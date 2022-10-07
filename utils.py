import pandas as pd
import datetime


def data_save(trainlist, validlist, nowTime, best_acc):
    df_train = pd.DataFrame(trainlist, columns=('epoch', 'T loss', 'T accu'))
    df_valid = pd.DataFrame(validlist, columns=('epoch', 'V loss', 'V accu'))
    df_valid = df_valid[df_valid.columns.difference(df_train.columns)]
    df = pd.concat([df_train,df_valid], axis=1)
    df.to_csv(str(nowTime)+'_'+'result'+'_'+str(best_acc)+'.csv')

    return