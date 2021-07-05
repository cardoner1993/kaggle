import pandas as pd
import sys

from train_config import config

sys.path.append('../common')
from DNN import DNN

test = pd.read_csv("../../data/test.csv")

## Pilar BertClassifier
# model = BertClassifier()
# model.load('../input/bertv1/bert_en_uncased/v1/')
# preds = model.predict(test["excerpt"])


# Todo move from model_name_datetime to model_name/datetime
model_config = config['models']['DNN']['dnn_conv']
model = DNN(logger=None, **model_config['parameters'])
model.load('../models/dnn_conv_20210629_104451')
preds = model.predict(test["excerpt"])


submission = pd.DataFrame({'id': test.id, 'target': preds})
submission.to_csv('/kaggle/working/submission.csv', index=False)