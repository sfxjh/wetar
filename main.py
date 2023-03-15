from copy import copy

from numpy.random import choice
from utils.info_regularizer import InfoNCE
from weakref import ProxyTypes
from tensorflow.python.keras.backend import identity
import torch
import os

from args import ProgramArgs, string_to_bool

args = ProgramArgs.parse(True)
args.build_environment()
args.build_logging()
args.build_logging_dir()

import logging
from typing import Union, Set

from textattack.models.tokenizers import GloveTokenizer
from textattack.models.helpers import GloveEmbeddingLayer
from tqdm import tqdm
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, KLDivLoss
from torch.optim import AdamW
# from transformers.optimization import AdamW
from torch.utils.data import ConcatDataset
import numpy as np

from data.reader import ClassificationReader, ClassificationReaderSpacy
from trainer import (
    BaseTrainer,
    FreeLBTrainer,
    HotflipTrainer,
    PGDTrainer,
    IBPTrainer,
    TokenAwareVirtualAdversarialTrainer,
    InfoBertTrainer,
    DNETrainer,
    MixUpTrainer,
    SAFERTrainer,
    MaskTrainer,
    ASCCTrainer,
    InstanceWisePGDTrainer,
    FeaturePairAdversarialTrainer,
    LearnToReweightTrainer,
    SMARTTrainer
)
from utils.config import LABEL_MAP, MODEL_CLASSES, DATASET_LABEL_NUM, GLOVE_CONFIGS, MODEL_LAYERS
from utils.metrics import Metric, ClassificationMetric, SimplifidResult
from utils.my_utils import convert_batch_to_bert_input_dict, replace_unk_by_indices, convert_dataset_to_batch
from utils.public import auto_create, check_and_create_path
from utils.textattack_utils import build_english_attacker, CustomTextAttackDataset
from utils.dne_utils import DecayAlphaHull, get_bert_vocab, WeightedEmbedding
from utils.ascc_utils import WarmupMultiStepLR
from data.instance import InputInstance

from textattack.loggers import AttackLogManager
from textattack.models.wrappers import HuggingFaceModelWrapper, PyTorchModelWrapper, HuggingFaceModelEnsembleWrapper
from textattack.augmentation.augmenter import Augmenter
from textattack.augmentation.faster_augmentor import FasterAugmenter
from textattack.transformations import WordSwapWordNet, WordSwapEmbedding, WordSwapMaskedLM
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import MaxWordsPerturbed
from model.models import LSTMModel, MixText, ASCCModel
from utils.certified import attacks, vocabulary, data_util
from utils.hook import ModuleHook, bert_model_hooker
from utils.datasets_loader import valid_batch_manager


class AttackBenchmarkTask(object):
    def __init__(self, args: ProgramArgs):
        self.methods = {'train': self.train,
                        'evaluate': self.evaluate,
                        'attack': self.attack,
                        'augment': self.augment,
                        'dev_eval': self.dev_aug_evaluate,
                        'attack_epochs': self.attack_epochs,
                        'eval_last': self.evaluate_last,
                        'get_adv_example': self.get_adv_examples,
                        'output': self.get_output_states,
                        }
        assert args.mode in self.methods, 'mode {} not found'.format(args.mode)

        self.tensor_input = False if args.training_type in args.type_accept_instance_as_input and args.mode == 'train' else True

        if args.model_type != 'lstm':
            self.tokenizer = self._build_tokenizer(args)
            self.dataset_reader = ClassificationReader(model_type=args.model_type, max_seq_len=args.max_seq_len)
        else:
            self.dataset_reader = ClassificationReaderSpacy(model_type=args.model_type,
                                                            max_seq_len=args.max_seq_len)
            # build attack surface
            if string_to_bool(args.use_lm):
                if args.dataset_name == 'imdb':
                    lm_file = args.imdb_lm_file
                elif args.dataset_name == 'snli':
                    lm_file = args.snli_lm_file
                else:
                    raise NotImplementedError
                self.attack_surface = auto_create(
                    f'{args.dataset_name}_attack_surface_lm',
                    lambda: attacks.LMConstrainedAttackSurface.from_files(args.neighbor_file, lm_file),
                    True, path=args.cache_path
                )
            else:
                self.attack_surface = auto_create(
                    'attack_surface_cf',
                    lambda: attacks.WordSubstitutionAttackSurface.from_file(args.neighbor_file),
                    True, path=args.cache_path
                )

        # if args.use_dev_aug == 'False':
        self.train_raw, self.eval_raw, self.test_raw = auto_create(
            f'{args.dataset_name}_raw_datasets', lambda: self._build_raw_dataset(args),
            True, path=args.cache_path
        )
        self.train_dataset, self.eval_dataset, self.test_dataset = auto_create(
            f'{args.dataset_name}_tokenized_datasets', lambda: self._build_tokenized_dataset(args),
            True, path=args.cache_path
        )
        # else:
        #     self.train_raw, self.eval_raw, self.test_raw = auto_create(
        #         f'{args.dataset_name}_dev_{args.dev_aug_attacker}_{args.dev_aug_ratio}_datasets', lambda: self._build_raw_dataset(args),
        #         True, path=args.cache_path
        #     )
        #     self.train_dataset, self.eval_dataset, self.test_dataset = auto_create(
        #             f'{args.dataset_name}_dev_{args.dev_aug_attacker}_{args.dev_aug_ratio}_tokenized_datasets', lambda: self._build_tokenized_dataset(args),
        #             True, path=args.cache_path
        #     )
        if not self.tensor_input:
            self.train_dataset = self.train_raw

        if args.model_type == 'lstm':
            word_set = self.dataset_reader.get_word_set(self.train_raw + self.eval_raw,
                                                        args.counter_fitted_file, self.attack_surface)
            self.vocab, self.word_mat = auto_create(
                f'{args.dataset_name}_glove_vocab_emb',
                lambda: vocabulary.Vocabulary.read_word_vecs(word_set, args.glove_dir, args.glove_name,
                                                             args.device, prepend_null=True),
                True,
                path=args.cache_path
            )

        self.data_loader, self.eval_data_loader, self.test_data_loader = self._build_dataloader(args)
        self.model = self._build_model(args)
        self.attack_model=self._build_model(args)
        self._build_estimator(args)
        self.forbidden_words = self._build_forbidden_words(args.sentiment_path) if string_to_bool(
            args.keep_sentiment_word) else None
        self.loss_function = self._build_criterion(args)

    def train(self, args: ProgramArgs):
        self.optimizer = self._build_optimizer(args)
        self.lr_scheduler = self._build_lr_scheduler(args)
        self.writer = self._build_writer(args)
        trainer = self._build_trainer(args)
        best_metric = None
        epoch_now = 0  # self._check_training_epoch(args)
        if args.training_type == 'l2rew':
            # pass
            iterator = tqdm(self.data_loader)
            for batch in iterator:
                batch = next(self.data_loader._get_iterator())
                try:
                    loss = self.trainer.base_train_batch(args, batch)
                    iterator.set_description('loss: {:.4f}'.format(loss))
                except RuntimeError as e:
                    raise e
            self._saving_model_by_epoch(args, epoch_now)
            metric = self.evaluate(args, is_training=True)
            epoch_now = 1
        for epoch_time in range(epoch_now, args.epochs):
            if args.training_type == 'ibp':
                trainer.set_epoch(epoch_time)
            if args.training_type == 'l2rew':
                if epoch_time % args.epochs_update_valid == 0 or epoch_time < 2:
                    self.epoch_data_augmenter(args)
                trainer.train_epoch(args, epoch_time, self.valid_rob_dataloader)
            else:
                trainer.train_epoch(args, epoch_time)

            # saving model according to epoch_time
            self._saving_model_by_epoch(args, epoch_time)

            if args.training_type == 'l2rew':
                self.trainer.save_global_weights(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                         args.build_saving_file_name(description='global_weights'))


            # evaluate model according to epoch_time
            metric = self.evaluate(args, is_training=True)
            if args.training_type == 'l2rew':
                rob_metric = self.evaluate_valid_rob(args)

            # update best metric
            # if best_metric is None, update it with epoch metric directly, otherwise compare it with epoch_metric
            if best_metric is None or metric > best_metric:
                best_metric = metric
                self._save_model_to_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                         args.build_saving_file_name(description='best'))

            if epoch_time == args.epochs - 1:
                self._save_model_to_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                         args.build_saving_file_name(description='last'))

        # if args.training_type == 'l2rew':
        #     self.trainer.save_global_weights(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
        #                                  args.build_saving_file_name(description='global_weights'))
        self.evaluate(args)

    def dev_aug_evaluate(self, args: ProgramArgs):
        self.optimizer = self._build_optimizer(args)
        self.lr_scheduler = self._build_lr_scheduler(args)
        self.writer = self._build_writer(args)
        best_metric = None
        for epoch_time in range(args.epochs):
            self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                          args.build_saving_file_name(description='epoch{}'.format(epoch_time)))
            metric = self.evaluate(args, is_training=True)
            if best_metric is None or metric > best_metric:
                best_metric = metric
                self._save_model_to_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                         args.build_saving_file_name(
                                             description=f"best_{args.dev_aug_attacker}_{args.dev_aug_ratio}"))
        self.evaluate(args)

    @torch.no_grad()
    def evaluate(self, args: ProgramArgs, is_training: bool = False) -> Metric:
        if is_training:
            logging.info('Using current modeling parameter to evaluate')
            epoch_iterator = tqdm(self.eval_data_loader)
        elif args.evaluation_model_type == 'best':
            if args.use_dev_aug == 'False':
                self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                              args.build_saving_file_name(description='best'))
            else:
                self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                              args.build_saving_file_name(
                                                  description=f"best_{args.dev_aug_attacker}_{args.dev_aug_ratio}"))
            epoch_iterator = tqdm(self.test_data_loader)
        else:
            if args.use_dev_aug == 'False':
                self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                              args.build_saving_file_name(description='last'))
            else:
                self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                              args.build_saving_file_name(
                                                  description=f"last_{args.dev_aug_attacker}_{args.dev_aug_ratio}"))
            epoch_iterator = tqdm(self.test_data_loader)
        if args.evaluation_data_type == 'dev':
            epoch_iterator == tqdm(self.eval_data_loader)
        self.model.eval()

        metric = ClassificationMetric(compare_key=args.compare_key)
        for step, batch in enumerate(epoch_iterator):
            if args.model_type == 'lstm':
                batch = tuple(t.to(args.device) for t in batch)
                golds = batch[3].long().t().squeeze()
                logits = self.model.forward(batch, compute_bounds=False)
                losses = self.loss_function(logits, golds)
            else:
                assert isinstance(batch[0], torch.Tensor)
                batch = tuple(t.to(args.device) for t in batch)
                golds = batch[3]
                inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
                logits = self.model.forward(**inputs)[0]
                losses = self.loss_function(logits.view(-1, DATASET_LABEL_NUM[args.dataset_name]), golds.view(-1))
                epoch_iterator.set_description('loss: {:.4f}'.format(torch.mean(losses)))
            metric(losses, logits, golds)

        print(metric)
        logging.info(metric)
        return metric

    @torch.no_grad()
    def evaluate_valid_rob(self, args: ProgramArgs, is_training: bool = False):
        logging.info('Using current modeling parameter to evaluate')
        epoch_iterator = tqdm(self.valid_rob_dataloader)
        self.model.eval()

        metric = ClassificationMetric(compare_key=args.compare_key)
        for step, batch in enumerate(epoch_iterator):
            if args.model_type == 'lstm':
                batch = tuple(t.to(args.device) for t in batch)
                golds = batch[3].long().t().squeeze()
                logits = self.model.forward(batch, compute_bounds=False)
                losses = self.loss_function(logits, golds)
            else:
                assert isinstance(batch[0], torch.Tensor)
                batch = tuple(t.to(args.device) for t in batch)
                golds = batch[3]
                inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
                logits = self.model.forward(**inputs)[0]
                losses = self.loss_function(logits.view(-1, DATASET_LABEL_NUM[args.dataset_name]), golds.view(-1))
                epoch_iterator.set_description('loss: {:.4f}'.format(torch.mean(losses)))
            metric(losses, logits, golds)

        print("rob ", metric)
        logging.info(metric)
        return metric

    @torch.no_grad()
    def evaluate_last(self, args: ProgramArgs, is_training: bool = False):
        if args.eval_epoch < 0:
            self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                      args.build_saving_file_name(description='last'))
        else:
            self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                      args.build_saving_file_name(description=f'epoch{args.eval_epoch}'))
        self.model.eval()
        if args.evaluation_data_type == 'dev':
            epoch_iterator = tqdm(self.eval_data_loader)
        else:
            epoch_iterator = tqdm(self.test_data_loader)

        metric = ClassificationMetric(compare_key=args.compare_key)
        for step, batch in enumerate(epoch_iterator):
            if args.model_type == 'lstm':
                batch = tuple(t.to(args.device) for t in batch)
                golds = batch[3].long().t().squeeze()
                logits = self.model.forward(batch, compute_bounds=False)
                losses = self.loss_function(logits, golds)
            else:
                assert isinstance(batch[0], torch.Tensor)
                batch = tuple(t.to(args.device) for t in batch)
                golds = batch[3]
                inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
                logits = self.model.forward(**inputs)[0]
                losses = self.loss_function(logits.view(-1, DATASET_LABEL_NUM[args.dataset_name]), golds.view(-1))
                epoch_iterator.set_description('loss: {:.4f}'.format(torch.mean(losses)))
            metric(losses, logits, golds)

        print(metric)
        logging.info(metric)
        return metric

    @torch.no_grad()
    def get_output_states(self, args: ProgramArgs):
        if args.attack_epoch_type == 'best':
            self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                          args.build_saving_file_name(description='best'))
        elif args.attack_epoch_type == 'last':
            self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                          args.build_saving_file_name(description='last'))
        else:
            raise NotImplementedError
        self.model.eval()
        bert_model_hooker(self.model, MODEL_LAYERS[args.model_type])
        if args.feature_dataset_use_base == "False":
            same_label_path = f'{args.saving_feature_dataset_path}' + args.build_feature_file_name('samelabel') + '.txt'
            original_path = f'{args.saving_feature_dataset_path}' + args.build_feature_file_name('original') + '.txt'
            diff_label_path = f'{args.saving_feature_dataset_path}' + args.build_feature_file_name('difflabel') + '.txt'
        else:
            same_label_path = "temp_feature/dataset/samelabel-base-ev_datadev.txt"
            original_path = "temp_feature/dataset/original-base-ev_datadev.txt"
            diff_label_path = "temp_feature/dataset/difflabel-base-ev_datadev.txt"
        path = [original_path, same_label_path, diff_label_path]
        name = ["original", "samelabel", "difflabel"]
        dict_features = {}
        word_embeddings_layer = self.model.get_input_embeddings()
        from data.dataset import ListDataset
        from torch.utils.data import DataLoader
        for i in range(len(path)):
            test_raw = self.dataset_reader.read_from_file(path[i], None)
            test_list = [[], []]
            for j in range(len(test_raw)):
                test_list[0].append(test_raw[j].text_a)
                test_list[1].append(test_raw[j].label)
            dataset = ListDataset(test_raw)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            epoch_iterator = tqdm(dataloader)
            ModuleHook.hook_clean_up()
            temp_logits = None
            temp_inputs = None
            for data, label in epoch_iterator:
                inputs = self.tokenizer(data, return_tensors='pt', padding=True, max_length=128)
                embeddings = word_embeddings_layer(inputs['input_ids'].to(self.model.device))
                cls_index = [ind for ind in range(1, embeddings.shape[1])]
                cls_index = torch.tensor(cls_index).to(self.model.device)
                temp_cls = torch.index_select(embeddings, dim=1, index=cls_index).view(-1, embeddings.size(-1))
                temp_cls = torch.mean(temp_cls, dim=0, keepdim=True)
                if temp_inputs is None:
                    temp_inputs = temp_cls
                else:
                    temp_inputs = torch.cat((temp_inputs, temp_cls), dim=0)
                for key in inputs.keys():
                    inputs[key] = inputs[key].to(self.model.device)
                logits = self.model(**inputs)[0]
                if temp_logits is None:
                    temp_logits = logits.clone()
                else:
                    temp_logits = torch.cat((temp_logits, logits), dim=0)
            layer_outputs = []
            layer_outputs.append(temp_inputs)
            for num_layers in range(len(ModuleHook.forward_value)):
                temp_tensor = None
                for batch_out in ModuleHook.forward_value[num_layers]:
                    cls_index = [ind for ind in range(1, batch_out.shape[1])]
                    cls_index = torch.tensor(cls_index).to(self.model.device)
                    temp_cls = torch.index_select(batch_out, dim=1, index=cls_index).view(-1, batch_out.size(-1))
                    temp_cls = torch.mean(temp_cls, dim=0, keepdim=True)
                    if temp_tensor is None:
                        temp_tensor = temp_cls
                    else:
                        temp_tensor = torch.cat((temp_tensor, temp_cls), dim=0)
                layer_outputs.append(temp_tensor)
            layer_outputs.append(temp_logits)
            dict_features[name[i]] = layer_outputs
        torch.save(dict_features, f'{args.saving_feature_path}' + args.build_feature_file_name() + '.pth')

    def epoch_data_augmenter(self, args: ProgramArgs):
        choice_num = int(args.augment_ratio * args.batch_size)
        if choice_num == 0:
            valid_instances = self.eval_raw
            instances = [x for x in valid_instances if len(x.text_a.split(' ')) > 4]
            valid_rob_dataset = self.dataset_reader.get_dataset(instances, self.tokenizer)
            self.valid_rob_dataloader = self.dataset_reader.get_dataset_loader(
                dataset=valid_rob_dataset,
                tokenized=True,
                batch_size=args.valid_batch_size,
                shuffle=False
            )
            return None
        if args.base_aug == 'False':
            self.model.eval()
            model_wrapper = HuggingFaceModelWrapper(self.model, self.tokenizer, batch_size=args.batch_size)
        else:
            self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                        args.attack_model_path,model='attack')
            self.attack_model.eval()
            model_wrapper = HuggingFaceModelWrapper(self.attack_model, self.tokenizer, batch_size=args.batch_size)
        from utils.textattack_utils import augmenter
        attacker = augmenter(args.augment_method, model_wrapper)
        valid_instances = self.eval_raw
        instances = [x for x in valid_instances if len(x.text_a.split(' ')) > 4]
        attacker_log_manager = AttackLogManager()
        # choice_num = int(args.augment_ratio * len(instances))
        # choice_instances = np.random.choice(instances, size=(choice_num,), replace=False)
        ori_instances, need_aug_instances, aug_num_batch = valid_batch_manager(instances, args.batch_size,
                                                                               args.augment_ratio)
        if len(need_aug_instances) > 0:
            dataset = CustomTextAttackDataset.from_instances(args.dataset_name, need_aug_instances,
                                                             self.dataset_reader.get_labels())
            results_iterable = attacker.attack_dataset(dataset)
            aug_instances = []
            for (result, instance) in tqdm(zip(results_iterable, need_aug_instances), total=len(need_aug_instances)):
                try:
                    adv_sentence = result.perturbed_text()
                    aug_instances.append(InputInstance.create_instance_with_perturbed_sentence(instance, adv_sentence))
                except:
                    continue
            aug_length = len(aug_instances)
            valid_raw_instance = []
            for i in range(len(ori_instances)):
                if i < len(ori_instances) - 1:
                    ori_instances[i] += aug_instances[aug_num_batch * i:aug_num_batch * (i + 1) % aug_length]
                else:
                    ori_instances[i] += aug_instances[aug_num_batch * i:]
                valid_raw_instance += ori_instances[i]
        else:
            valid_raw_instance = ori_instances
        # valid_raw_instance = valid_instances + aug_instances
        valid_rob_dataset = self.dataset_reader.get_dataset(valid_raw_instance, self.tokenizer)
        self.valid_rob_dataloader = self.dataset_reader.get_dataset_loader(
            dataset=valid_rob_dataset,
            tokenized=True,
            batch_size=args.valid_batch_size,
            shuffle=False
        )
        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()

    def attack(self, args: ProgramArgs):
        if args.use_dev_aug == 'False':
            if args.attack_epoch_type == 'best':
                self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                              args.build_saving_file_name(description='best'))
            elif args.attack_epoch_type == 'last':
                self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                              args.build_saving_file_name(description='last'))
            else:
                raise NotImplementedError
        else:
            self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                          args.build_saving_file_name(
                                              description=f"best_{args.dev_aug_attacker}_{args.dev_aug_ratio}"))
        self.model.eval()
        attacker = self._build_attacker(args)

        if args.evaluation_data_type == 'dev':
            dataset = self.eval_raw
        elif args.evaluation_data_type == 'test':
            dataset = self.test_raw
        elif args.evaluation_data_type == 'train':
            dataset = self.train_raw
        test_instances = dataset

        attacker_log_path = '{}'.format(args.build_logging_path())
        attacker_log_path = os.path.join(f"{args.workspace}/log/{args.dataset_name}_{args.model_type}",
                                         attacker_log_path)
        attacker_log_manager = AttackLogManager()
        # attacker_log_manager.enable_stdout()
        attacker_log_manager.add_output_file(
            os.path.join(attacker_log_path,
                         f'{args.attack_method}_{args.neighbour_vocab_size}_{args.modify_ratio}_{args.attack_epoch_type}.txt'))
        test_instances = [x for x in test_instances if len(x.text_a.split(' ')) > 4]
        if not test_instances[0].label.isdigit():
            for i in range(len(test_instances)):
                test_instances[i].label = LABEL_MAP['nli'][test_instances[i].label]
        # attack multiple times for average success rate
        for i in range(args.attack_times):
            print("Attack time {}".format(i))
            total_len = len(test_instances)
            choise_num = min(total_len, args.attack_numbers)
            choice_instances = np.random.choice(test_instances, size=(choise_num,), replace=False)
            dataset = CustomTextAttackDataset.from_instances(args.dataset_name, choice_instances,
                                                             self.dataset_reader.get_labels())
            results_iterable = attacker.attack_dataset(dataset)
            description = tqdm(results_iterable, total=len(choice_instances))
            result_statistics = SimplifidResult()
            for result in description:
                try:
                    attacker_log_manager.log_result(result)
                    result_statistics(result)
                    description.set_description(result_statistics.__str__())
                except Exception as e:
                    print('error in process')
                    continue
        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()

    def attack_epochs(self, args: ProgramArgs):
        self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                      args.build_saving_file_name(description='epoch{}'.format(args.attack_epoch)))
        self.model.eval()
        attacker = self._build_attacker(args)

        if args.evaluation_data_type == 'dev':
            dataset = self.eval_raw
        else:
            dataset = self.test_raw
        test_instances = dataset

        attacker_log_path = '{}'.format(args.build_logging_path())
        attacker_log_path = os.path.join(f"{args.workspace}/log/{args.dataset_name}_{args.model_type}",
                                         attacker_log_path)
        attacker_log_manager = AttackLogManager()
        # attacker_log_manager.enable_stdout()
        attacker_log_manager.add_output_file(
            os.path.join(attacker_log_path,
                         f'{args.attack_method}_{args.neighbour_vocab_size}_{args.modify_ratio}_{args.attack_epoch}.txt'))
        test_instances = [x for x in test_instances if len(x.text_a.split(' ')) > 4]
        # attack multiple times for average success rate
        for i in range(args.attack_times):
            print("Attack time {}".format(i))
            total_len = len(test_instances)
            choise_num = min(total_len, args.attack_numbers)
            choice_instances = np.random.choice(test_instances, size=(choise_num,), replace=False)
            dataset = CustomTextAttackDataset.from_instances(args.dataset_name, choice_instances,
                                                             self.dataset_reader.get_labels())
            results_iterable = attacker.attack_dataset(dataset)
            description = tqdm(results_iterable, total=len(choice_instances))
            result_statistics = SimplifidResult()
            for result in description:
                try:
                    attacker_log_manager.log_result(result)
                    result_statistics(result)
                    description.set_description(result_statistics.__str__())
                except Exception as e:
                    print('error in process')
                    continue

        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()

    def augment(self, args: ProgramArgs):
        self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                      args.build_saving_file_name(description=args.attack_epoch_type))
        self.model.eval()
        attacker = self._build_attacker(args)
        training_instance = [instance for instance in self.train_raw if instance.length() > 4]
        training_len = len(training_instance)
        print('Training Set: {} sentences. '.format(training_len))
        attacker_log_manager = AttackLogManager()
        dataset = CustomTextAttackDataset.from_instances(f'{args.dataset_name}_aug', training_instance,
                                                         self.dataset_reader.get_labels())
        results_iterable = attacker.attack_dataset(dataset)
        aug_instances = []
        for (result, instance) in tqdm(zip(results_iterable, training_instance), total=training_len):
            try:
                adv_sentence = result.perturbed_text()
                aug_instances.append(InputInstance.create_instance_with_perturbed_sentence(instance, adv_sentence))
            except:
                continue
        instances = aug_instances + self.train_raw
        self.dataset_reader.saving_instances(instances, args.dataset_path, 'aug_{}'.format(args.attack_method))
        print(f'Augmented {len(aug_instances)} sentences. ')
        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()

    def get_adv_examples(self, args: ProgramArgs):
        if args.attack_epoch_type == 'last':
            self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                          args.build_saving_file_name(description="last"))
        elif args.attack_epoch_type == 'best':
            self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                          args.build_saving_file_name(description="best"))
        self.model.eval()
        attacker = self._build_attacker(args)
        if args.evaluation_data_type == 'test':
            origin_instance = [instance for instance in self.test_raw if instance.length() > 4]
        elif args.evaluation_data_type == 'dev':
            origin_instance = [instance for instance in self.eval_raw if instance.length() > 4]
        instance_len = len(origin_instance)
        print('{} set: {} sentences. '.format(args.evaluation_data_type, instance_len))
        attacker_log_manager = AttackLogManager()
        dataset = CustomTextAttackDataset.from_instances(
            f'{args.dataset_name}_{args.evaluation_data_type}_aug', origin_instance, self.dataset_reader.get_labels())
        results_iterable = attacker.attack_dataset(dataset)
        original_ = []
        aug_instances = []
        label_changed_instances = []
        for (result, instance) in tqdm(zip(results_iterable, origin_instance), total=instance_len):
            try:
                adv_sentence = result.perturbed_result.adversarial_result.attacked_text.text
                label_changed_sentence = result.perturbed_text()
                if len(instance.text_a.split(' ')) != len(adv_sentence.split(' ')):
                    continue
                original_.append(instance)
                aug_instances.append(InputInstance.create_instance_with_perturbed_sentence(instance, adv_sentence))
                label_changed_instances.append(
                    InputInstance.create_instance_with_perturbed_sentence(instance, label_changed_sentence))
            except Exception as e:
                continue
        with open(f'{args.saving_feature_dataset_path}' + args.build_feature_file_name('samelabel') + '.txt', 'w',
                  encoding='utf-8') as f0:
            for instance in aug_instances:
                f0.write(f'{instance.text_a}\t{instance.label}\n')
        with open(f'{args.saving_feature_dataset_path}' + args.build_feature_file_name('original') + '.txt', 'w',
                  encoding='utf-8') as f1:
            for instance in original_:
                f1.write(f'{instance.text_a}\t{instance.label}\n')
        with open(f'{args.saving_feature_dataset_path}' + args.build_feature_file_name('difflabel') + '.txt', 'w',
                  encoding='utf-8') as f2:
            for instance in label_changed_instances:
                f2.write(f'{instance.text_a}\t{instance.label}\n')
        print(f'Augmented {len(aug_instances)} sentences.')
        pass

    def _save_model_to_file(self, save_dir: str, file_name: str):
        save_file_name = '{}.pth'.format(file_name)
        check_and_create_path(save_dir)
        save_path = os.path.join(save_dir, save_file_name)
        torch.save(self.model.state_dict(), save_path)
        logging.info('Saving model to {}'.format(save_path))

    def _saving_model_by_epoch(self, args: ProgramArgs, epoch: int):
        # saving
        if args.saving_step is not None and args.saving_step != 0:
            if (epoch - 1) % args.saving_step == 0:
                self._save_model_to_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                         args.build_saving_file_name(description='epoch{}'.format(epoch)))

    def _check_training_epoch(self, args: ProgramArgs):
        epoch_now = 0
        save_dir = f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}"
        for epoch in range(args.epochs):
            file_name = args.build_saving_file_name(description='epoch{}'.format(epoch))
            save_file_name = '{}.pth'.format(file_name)
            check_and_create_path(save_dir)
            save_path = os.path.join(save_dir, save_file_name)
            if os.path.exists(save_path) and os.path.isfile(save_path):
                epoch_now = epoch + 1
                continue
            else:
                if epoch_now != 0:
                    file_name = args.build_saving_file_name(description='epoch{}'.format(epoch - 1))
                    self._loading_model_from_file(save_dir, file_name)
                break
        return epoch_now

    def _loading_model_from_file(self, save_dir: str, file_name: str,model='model'):
        load_file_name = '{}.pth'.format(file_name)
        load_path = os.path.join(save_dir, load_file_name)
        assert os.path.exists(load_path) and os.path.isfile(load_path), '{} not exits'.format(load_path)
        if(model=='model'):
            self.model.load_state_dict(torch.load(load_path), strict=False)
            logging.info('Loading model from {}'.format(load_path))
        elif(model=='attack'):
            self.attack_model.load_state_dict(torch.load(load_path), strict=False)
            logging.info('Loading attack model from {}'.format(load_path))
        else:
            raise NotImplementedError

    def _build_trainer(self, args: ProgramArgs):
        trainer = BaseTrainer(self.data_loader, self.model, self.loss_function, self.optimizer, self.lr_scheduler,
                              self.writer, args.epochs)
        if args.training_type == 'freelb':
            trainer = FreeLBTrainer(self.data_loader, self.model, self.loss_function, self.optimizer, self.lr_scheduler,
                                    self.writer)
        elif args.training_type == 'pgd':
            trainer = PGDTrainer(self.data_loader, self.model, self.loss_function, self.optimizer, self.lr_scheduler,
                                 self.writer)
        elif args.training_type == 'advhotflip':
            trainer = HotflipTrainer(args, self.tokenizer, self.data_loader, self.model, self.loss_function,
                                     self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'ibp':
            trainer = IBPTrainer(args, self.data_loader, self.model, self.loss_function, self.optimizer,
                                 self.lr_scheduler, self.writer)
        elif args.training_type == 'tavat':
            trainer = TokenAwareVirtualAdversarialTrainer(args, self.data_loader, self.model, self.loss_function,
                                                          self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'infobert':
            trainer = InfoBertTrainer(args, self.data_loader, self.model, self.loss_function,
                                      self.optimizer, self.lr_scheduler, self.writer, self.mi_upper_estimator,
                                      self.mi_estimator)
        elif args.training_type == 'dne':
            trainer = DNETrainer(args, self.data_loader, self.model, self.loss_function,
                                 self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'mixup':
            trainer = MixUpTrainer(args, self.data_loader, self.model, self.loss_function,
                                   self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'safer':
            trainer = SAFERTrainer(args, self.tokenizer, self.dataset_reader, self.data_loader, self.model,
                                   self.loss_function, self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'mask':
            trainer = MaskTrainer(args, self.tokenizer, self.dataset_reader, self.data_loader, self.model,
                                  self.loss_function, self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'ascc':
            trainer = ASCCTrainer(args, self.data_loader, self.model, self.loss_function,
                                  self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'iwpgd':
            trainer = InstanceWisePGDTrainer(args, self.data_loader, self.model, self.loss_function, self.optimizer,
                                             self.lr_scheduler,
                                             self.writer)
        elif args.training_type == 'featurepair':
            trainer = FeaturePairAdversarialTrainer(args, self.data_loader, self.model, self.loss_function,
                                                    self.optimizer, self.lr_scheduler,
                                                    self.writer, self.estimator_list)
        elif args.training_type == 'l2rew':
            trainer = LearnToReweightTrainer(args, self.data_loader, self.model, self.loss_function, self.optimizer,
                                             self.lr_scheduler, self.writer, len(self.train_raw))
        elif args.training_type == 'smart':
            trainer = SMARTTrainer(args,self.data_loader,self.model,self.loss_function,self.optimizer,self.lr_scheduler,self.writer)
        self.trainer = trainer
        return trainer

    def _build_optimizer(self, args: ProgramArgs, **kwargs):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        if args.training_type == 'infobert':
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)] +
                              list(self.mi_estimator.parameters()),
                    "weight_decay": args.weight_decay,
                },
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
        elif args.training_type == 'featurepair':
            esitimator_parameters = []
            for i in range(len(self.estimator_list)):
                esitimator_parameters += list(self.estimator_list[i].parameters())
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)] +
                              esitimator_parameters,
                    "weight_decay": args.weight_decay,
                },
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        return optimizer

    def _build_proxy(self, args: ProgramArgs):
        config_class, model_class, _ = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(
            args.model_name_or_path,
            num_labels=DATASET_LABEL_NUM[args.dataset_name],
            finetuning_task=args.dataset_name,
            output_hidden_states=True,
        )
        proxy = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool('ckpt' in args.model_name_or_path),
            config=config
        ).to(args.device)
        if not hasattr(args, 'proxy_opt') or args.proxy_opt == 'adam':
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in proxy.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {"params": [p for n, p in proxy.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            proxy_opt = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        elif args.proxy_opt == 'sgd':
            from torch.optim.sgd import SGD
            proxy_opt = SGD(proxy.parameters(), lr=args.learning_rate)
        from transformers import get_linear_schedule_with_warmup
        proxy_scheduler = get_linear_schedule_with_warmup(proxy_opt, num_warmup_steps=args.warmup_steps,
                                                          num_training_steps=len(
                                                              self.train_dataset) // args.batch_size * args.epochs)
        return proxy, proxy_opt, proxy_scheduler

    def _build_model(self, args: ProgramArgs):
        if args.model_type == 'lstm':
            model = LSTMModel(
                GLOVE_CONFIGS[args.glove_name]['size'], args.hidden_size,
                self.word_mat, args.device,
                num_labels=DATASET_LABEL_NUM[args.dataset_name],
                pool=args.pool,
                dropout=args.dropout_prob,
                no_wordvec_layer=args.no_wordvec_layer).to(args.device)
        elif args.training_type == 'mixup':
            config_class, _, _ = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(
                args.model_name_or_path,
                num_labels=DATASET_LABEL_NUM[args.dataset_name],
                finetuning_task=args.dataset_name,
                output_hidden_states=True,
            )
            model = MixText.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('ckpt' in args.model_name_or_path),
                config=config
            ).to(args.device)
        elif args.training_type == 'dne':
            config_class, model_class, _ = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(
                args.model_name_or_path,
                num_labels=DATASET_LABEL_NUM[args.dataset_name],
                finetuning_task=args.dataset_name,
                output_hidden_states=True,
            )
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('ckpt' in args.model_name_or_path),
                config=config
            ).to(args.device)
            bert_vocab = get_bert_vocab()
            hull = DecayAlphaHull.build(
                alpha=args.dir_alpha,
                decay=args.dir_decay,
                nbr_file=args.nbr_file,
                vocab=bert_vocab,
                nbr_num=args.nbr_num,
                second_order=True
            )
            # here we just focus on bert model
            model.bert.embeddings.word_embeddings = WeightedEmbedding(
                num_embeddings=bert_vocab.get_vocab_size('tokens'),
                embedding_dim=768,
                padding_idx=model.bert.embeddings.word_embeddings.padding_idx,
                _weight=model.bert.embeddings.word_embeddings.weight,
                hull=hull,
                sparse=False)
        elif args.training_type == 'ascc':
            config_class, _, _ = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(
                args.model_name_or_path,
                num_labels=DATASET_LABEL_NUM[args.dataset_name],
                finetuning_task=args.dataset_name,
                output_hidden_states=True,
            )
            model = ASCCModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('ckpt' in args.model_name_or_path),
                config=config
            ).to(args.device)
            bert_vocab = get_bert_vocab()
            model.build_nbrs(args.nbr_file, bert_vocab, args.alpha, args.num_steps)
        elif args.training_strategy == "lp":
            config_class, model_class, _ = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(
                args.model_name_or_path,
                num_labels=DATASET_LABEL_NUM[args.dataset_name],
                finetuning_task=args.dataset_name,
                output_hidden_states=True,
            )
            from model.bert_with_linears import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('ckpt' in args.model_name_or_path),
                config=config
            ).to(args.device)
        else:
            config_class, model_class, _ = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(
                args.model_name_or_path,
                num_labels=DATASET_LABEL_NUM[args.dataset_name],
                finetuning_task=args.dataset_name,
                output_hidden_states=True,
            )
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('ckpt' in args.model_name_or_path),
                config=config
            ).to(args.device)
        return model

    def _build_estimator(self, args):
        from utils.info_regularizer import CLUB, InfoNCE
        hidden_size = self.model.config.hidden_size
        if args.training_type == 'infobert':
            self.mi_upper_estimator = CLUB(hidden_size, hidden_size, beta=args.beta).to(self.model.device)
            self.mi_estimator = InfoNCE(hidden_size, hidden_size).to(self.model.device)
        if args.training_type == 'featurepair':
            self.estimator_list = []
            for i in range(args.layer_num):
                self.estimator_list.append(InfoNCE(hidden_size, hidden_size).to(self.model.device))
        return None

    def _build_tokenizer(self, args: ProgramArgs):
        _, _, tokenizer_class = MODEL_CLASSES[args.model_type]
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=string_to_bool(args.do_lower_case)
        )
        return tokenizer

    def _build_raw_dataset(self, args: ProgramArgs):
        train_raw = self.dataset_reader.read_from_file(file_path=f"{args.workspace}/dataset/{args.dataset_name}",
                                                       split='train')
        if args.use_dev_aug == 'False':
            eval_raw = self.dataset_reader.read_from_file(file_path=f"{args.workspace}/dataset/{args.dataset_name}",
                                                          split='dev')
        else:
            eval_raw = self.dataset_reader.read_from_file(file_path=f"{args.workspace}/dataset/{args.dataset_name}",
                                                          split=f"dev_{args.dev_aug_attacker}_{args.dev_aug_ratio}")
        test_raw = self.dataset_reader.read_from_file(file_path=f"{args.workspace}/dataset/{args.dataset_name}",
                                                      split='test')

        return train_raw, eval_raw, test_raw

    def _build_tokenized_dataset(self, args: ProgramArgs):
        assert isinstance(self.dataset_reader, ClassificationReader)
        train_dataset = self.dataset_reader.get_dataset(self.train_raw, self.tokenizer)
        eval_dataset = self.dataset_reader.get_dataset(self.eval_raw, self.tokenizer)
        test_dataset = self.dataset_reader.get_dataset(self.test_raw, self.tokenizer)

        return train_dataset, eval_dataset, test_dataset

    def _build_dataloader(self, args: ProgramArgs):
        if args.model_type == 'lstm':
            train_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.train_dataset,
                                                                       batch_size=args.batch_size,
                                                                       vocab=self.vocab)
            eval_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.eval_dataset,
                                                                      batch_size=args.batch_size,
                                                                      vocab=self.vocab)
            test_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.test_dataset,
                                                                      batch_size=args.batch_size,
                                                                      vocab=self.vocab)
        else:
            assert isinstance(self.dataset_reader, ClassificationReader)
            if string_to_bool(args.use_aug):
                aug_raw = auto_create(
                    f'{args.dataset_name}_raw_aug_{args.aug_attacker}',
                    lambda: self.dataset_reader.read_from_file(
                        file_path=f"{args.workspace}/dataset/{args.dataset_name}",
                        split=f'aug_{args.aug_attacker}'),
                    True,
                    path=args.cache_path
                )
                aug_dataset = auto_create(
                    f'{args.dataset_name}_tokenized_aug_{args.aug_attacker}',
                    lambda: self.dataset_reader.get_dataset(aug_raw, self.tokenizer),
                    True,
                    path=args.cache_path
                )

                if args.aug_ratio == 1.0:
                    train_data_loader = self.dataset_reader.get_dataset_loader(dataset=aug_dataset,
                                                                               tokenized=True,
                                                                               batch_size=args.batch_size,
                                                                               shuffle=string_to_bool(args.shuffle))
                elif args.aug_ratio == 0.5:
                    train_data_loader = self.dataset_reader.get_dataset_loader(
                        dataset=ConcatDataset([self.train_dataset, aug_dataset]),
                        tokenized=True,
                        batch_size=args.batch_size,
                        shuffle=string_to_bool(args.shuffle))
                else:
                    raise NotImplementedError
                    train_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.train_dataset,
                                                                               tokenized=True,
                                                                               batch_size=args.batch_size,
                                                                               shuffle=string_to_bool(args.shuffle))
            else:
                train_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.train_dataset,
                                                                           tokenized=self.tensor_input,
                                                                           batch_size=args.batch_size,
                                                                           shuffle=string_to_bool(args.shuffle))
            eval_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.eval_dataset,
                                                                      tokenized=True,
                                                                      batch_size=args.batch_size,
                                                                      shuffle=string_to_bool(args.shuffle))
            test_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.test_dataset,
                                                                      tokenized=True,
                                                                      batch_size=args.batch_size,
                                                                      shuffle=string_to_bool(args.shuffle))
        return train_data_loader, eval_data_loader, test_data_loader

    def _build_criterion(self, args: ProgramArgs):
        # if args.attack_method == 'trades':
        #     return KLDivLoss(reduction='batchmean')
        # if args.model_type == 'lstm':
        #     return BCEWithLogitsLoss(reduction='mean')
        return CrossEntropyLoss(reduction='none')

    # def _build_lr_scheduler(self, args: ProgramArgs):
    #     if args.training_type == 'ascc':
    #         return WarmupMultiStepLR(self.optimizer, (40, 80), 0.1, 1.0 / 10.0, 2, 'linear')
    #     return CosineAnnealingLR(self.optimizer, len(self.train_dataset) // args.batch_size * args.epochs)
    def _build_lr_scheduler(self, args: ProgramArgs):
        if args.training_type == 'ascc':
            return WarmupMultiStepLR(self.optimizer, (40, 80), 0.1, 1.0 / 10.0, 2, 'linear')
        if args.scheduler == 'cos':
            return CosineAnnealingLR(self.optimizer, len(self.train_dataset) // args.batch_size * args.epochs)
        elif args.scheduler == 'slanted':
            from allennlp.training.learning_rate_schedulers.slanted_triangular import SlantedTriangular
            return SlantedTriangular(self.optimizer, num_epochs=args.epochs)
        elif args.scheduler == 'linear':
            from transformers import get_linear_schedule_with_warmup
            return get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_steps,
                                                   num_training_steps=len(
                                                       self.train_dataset) // args.batch_size * args.epochs)

    def _build_writer(self, args: ProgramArgs, **kwargs) -> Union[SummaryWriter, None]:
        writer = None
        if args.tensorboard == 'yes':
            tensorboard_file_name = '{}-tensorboard'.format(args.build_logging_path())
            tensorboard_path = os.path.join(f"{args.workspace}/log/{args.dataset_name}_{args.model_type}",
                                            tensorboard_file_name)
            writer = SummaryWriter(tensorboard_path)
        return writer

    def _build_forbidden_words(self, file_path: str) -> Set[str]:
        sentiment_words_set = set()
        with open(file_path, 'r', encoding='utf8') as file:
            for line in file.readlines():
                sentiment_words_set.add(line.strip())
        return sentiment_words_set

    def _build_attacker(self, args: ProgramArgs):
        if args.training_type in ['dne', 'safer', 'mask']:
            model_wrapper = HuggingFaceModelEnsembleWrapper(args, self.model, self.tokenizer)
        elif args.model_type != 'lstm':
            model_wrapper = HuggingFaceModelWrapper(self.model, self.tokenizer, batch_size=args.batch_size)
        else:
            tokenizer = GloveTokenizer(word_id_map=self.vocab.word2index,
                                       unk_token_id=0,
                                       pad_token_id=1,
                                       max_length=args.max_seq_len
                                       )
            model_wrapper = PyTorchModelWrapper(self.model, tokenizer, batch_size=args.batch_size)

        attacker = build_english_attacker(args, model_wrapper)
        return attacker


if __name__ == '__main__':
    logging.info(args)
    save_dir=f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}"
    save_file_name="{}.pth".format(args.build_saving_file_name(description='epoch{}'.format(9)))
    save_path=os.path.join(save_dir, save_file_name)
    # if(args.mode=='train' and os.path.exists(save_path)):
    #     print(save_path)
    #     quit(0)
    attacker_log_path = '{}'.format(args.build_logging_path())
    attacker_log_path = os.path.join(f"{args.workspace}/log/{args.dataset_name}_{args.model_type}",
                                         attacker_log_path)
    attacker_log_path=os.path.join(attacker_log_path,
                         f'{args.attack_method}_{args.neighbour_vocab_size}_{args.modify_ratio}_{args.attack_epoch_type}.txt')
    print(attacker_log_path)
    # if(args.mode=='attack' and os.path.exists(attacker_log_path)):
    #     print(attacker_log_path)
    #     quit(1)
    test = AttackBenchmarkTask(args)

    test.methods[args.mode](args)
