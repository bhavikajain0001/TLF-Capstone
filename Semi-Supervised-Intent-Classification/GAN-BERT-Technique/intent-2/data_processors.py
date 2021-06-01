## data_processors.py

import os
import csv
import tensorflow as tf
import tokenization
from tokenization import convert_to_unicode


class InputExample(object):

    """A single training/test example for simple sequence classification."""

    def __init__(
        self,
        guid,
        text_a,
        text_b=None,
        label=None,
        ):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):

    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):

    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_id,
        label_mask=None,
        is_real_example=True,
        ):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example
        self.label_mask = label_mask


class DataProcessor(object):

    """Base class for data converters for sequence classification data sets."""

    def get_labeled_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""

        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""

        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for test."""

        raise NotImplementedError()

    def get_validation_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""

        raise NotImplementedError()

    def get_OOS_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for out of space samples."""

        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""

        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""

        with tf.gfile.Open(input_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t', quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class QcFineProcessor(DataProcessor):

    """Processor for the MultiNLI data set (GLUE version)."""

    def get_labeled_examples(self, data_dir):
        """See base class."""

        return self._create_examples(os.path.join(data_dir,
                'labeled_Q2.tsv'), 'train')

    def get_unlabeled_examples(self, data_dir):
        """See base class."""

        return self._create_examples(os.path.join(data_dir,
                'unlabeled_Q2.tsv'), 'train')

    def get_test_examples(self, data_dir):
        """See base class."""

        return self._create_examples(os.path.join(data_dir, 'test_Q2.tsv'
                ), 'test')

    def get_labels(self, data_dir, set_type):
        """See base class."""
        return ['Explore_params', 'Effect_ofparams', 'Dependencies_bwparams','Selected_params', 'Monitoring_objectives','Tradeoff_bw_objectives']

    def _create_examples(self, input_file, set_type):
        """Creates examples for the training and dev sets."""

        examples = []

        with open(input_file, 'r') as f:
            contents = f.read()
            file_as_list = contents.splitlines()
            for line in file_as_list[1:]:
                split = line.split(' ')
                question = ' '.join(split[1:])

                guid = '%s-%s' % (set_type,
                                  tokenization.convert_to_unicode(line))
                text_a = tokenization.convert_to_unicode(question)
                inn_split = split[0].split(':')
                label = inn_split[0]
                examples.append(InputExample(guid=guid, text_a=text_a,
                                text_b=None, label=label))
            f.close()

        return examples


class GeneralProcessor(DataProcessor):

    """Processor for the MultiNLI data set (GLUE version)."""

    def get_labeled_examples(self, data_dir):
        """See base class."""

        (examples, _) = self._create_examples(os.path.join(data_dir,
                'train_Q2.tsv'), 'train')
        return examples

    def get_unlabeled_examples(self, data_dir):
        """See base class."""

        (examples, _) = self._create_examples(os.path.join(data_dir,
                'unlabeled_Q2.tsv'), 'train')
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""

        (examples, _) = self._create_examples(os.path.join(data_dir,
                'test_Q2.tsv'), 'test')
        return examples

    def get_validation_examples(self, data_dir):
        """See base class."""

        (examples, _) = self._create_examples(os.path.join(data_dir,
                'valid_Q2.tsv'), 'val')
        return examples

    def get_OOS_test_examples(self, data_dir):
        """See base class."""

        (examples, _) = self._create_examples(os.path.join(data_dir,
                'test_OOS_Q2.tsv'), 'test')
        return examples

    def get_labels(self, data_dir, set_type='gan_bert'):
        """See base class."""

        (_, unique_label_list) = \
            self._create_examples(os.path.join(data_dir, 'train_Q2.tsv'),
                                  'train')
        if set_type == 'gan_bert':
            unique_label_list.insert(0,'UNK_UNK')
            unique_label_list.insert(0,'OOS')

        print ()
        print ('======================================================================================')
        print ('                                     Unique Label List                                ')
        print (unique_label_list)
        print ('======================================================================================')
        print ()
        return unique_label_list

    def _create_examples(self, input_file, set_type):
        """Creates examples for the training and dev sets."""

        examples = []
        unique_label_list = []

        with open(input_file, 'r') as f:
            contents = f.read()
            file_as_list = contents.splitlines()
            for line in file_as_list[1:]:
                split = line.split(' ')
                question = ' '.join(split[1:])

                guid = '%s-%s' % (set_type,
                                  tokenization.convert_to_unicode(line))
                text_a = tokenization.convert_to_unicode(question)
                label = split[0]
                if label not in unique_label_list:
                    unique_label_list.append(label)
                examples.append(InputExample(guid=guid, text_a=text_a,
                                text_b=None, label=label))
            f.close()

        return (examples, unique_label_list)
