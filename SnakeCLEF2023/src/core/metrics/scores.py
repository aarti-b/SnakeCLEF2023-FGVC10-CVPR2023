import numpy as np
import torch
from sklearn.metrics import (f1_score, accuracy_score, top_k_accuracy_score,)

__all__ = ['classification_scores']

def check_venomous(arr):
    ls=[741, 857, 1736, 1118, 415, 154, 1530, 431, 1317, 1190, 159, 
                224, 205, 1035, 221, 228, 412, 238, 1076, 1107, 113, 562, 1072, 1119, 237, 216, 248, 225, 429,
                    512, 1052, 443, 1714, 1066, 1749, 1734, 33, 421, 27, 1081, 601, 462, 600, 1093, 739, 1739, 1746, 
                    423, 1316, 211, 425, 223, 1073, 806, 1393, 1671, 1670, 442, 464, 1742, 218, 1528, 1737, 417, 1703, 3, 
                    1106, 1230, 1056, 247, 1452, 1650, 430, 1037, 1684, 437, 1125, 424, 1745, 427, 0, 301, 783, 4, 1121, 1115, 
                    1688, 1713, 1315, 1053, 5, 1669, 160, 112, 308, 1071, 433, 1735, 1094, 410, 214, 743, 243, 2, 419, 307, 233, 
                    6, 858, 1191, 151, 1651, 1686, 310, 245, 595, 596, 407, 405, 1090, 439, 1040, 409, 440, 434, 403, 1059, 1082, 1652, 
                    416, 1114, 1065, 1044, 1373, 110, 242, 1034, 511, 1068, 829, 1116, 1129, 222, 1666, 830, 422, 285, 828, 411, 257,
                    1192, 1088, 280, 420, 1370, 1391, 1042, 1126, 1113, 804, 510, 1041, 1529, 1158, 207, 1318, 1117, 406, 414, 209,
                    1313, 1668, 1060, 1108, 1046, 259, 208, 1112, 213, 1390, 1395, 1389, 1124, 309, 1063, 256, 1747, 418, 1376, 260, 
                    1653, 1109, 402, 1128, 1050, 279, 1127, 302, 215, 428, 1319, 1329, 599, 1120, 206, 1122, 1327, 470, 236, 1665, 597,
                    235, 1396, 232, 32, 152, 1330, 1061, 1674, 999, 1036, 29, 204, 855, 241, 261, 262, 1679, 226, 1662, 240, 1687, 229, 
                    1685, 1744, 1741, 1110, 1057, 805, 219, 244, 1328, 463, 413, 1663, 513, 1740, 1392, 598, 1123, 856, 249, 1048, 34, 
                    1382, 1383, 1033, 1, 234, 740, 1375, 465, 162, 1079, 212, 1049, 258, 30, 1229, 1087, 1682, 150, 1312, 1680, 1743,
                        114, 1314, 742, 1371, 217, 1091, 210, 1372, 1374, 1689]
    return np.isin(arr, ls).astype(int)


def SnakeLossMetric(y_pred, y_true):
    l_sum=0
    a1 = (torch.tensor(y_pred) == torch.tensor(y_true)).type(torch.int).detach().numpy()
    a2 = check_venomous(y_pred)
    a3 = check_venomous(y_true)

    for i in range(len(a1)):
        if a1[i]==1:
            l_sum+=0
        elif (a1[i]==0 and a2[i]==0 and a3[i]==0):
            l_sum+=1
        elif (a1[i]==0 and a2[i]==0 and a3[i]==1): 
            l_sum+=2
        elif (a1[i]==0 and a2[i]==1 and a3[i]==1):
            l_sum+=2
        elif (a1[i]==0 and a2[i]==1 and a3[i]==0):
            l_sum+=5

    return l_sum
def classification_scores(preds, targs, *, top_k=3):
    preds_argmax = preds.argmax(1)
    labels = np.arange(preds.shape[1])
    scores = {}
    scores['accuracy'] = accuracy_score(targs, preds_argmax)
    scores['snake_score']=SnakeLossMetric(preds_argmax, targs)
    if top_k is not None and preds.shape[1] > 2:
        scores[f'top_{top_k}'] = top_k_accuracy_score(targs, preds, k=top_k, labels=labels)
    scores['f1_score'] = f1_score(targs, preds_argmax, labels=labels, average='macro', zero_division=0)
    
    return scores


