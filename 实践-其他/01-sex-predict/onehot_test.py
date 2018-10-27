from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# onehot
ohe = OneHotEncoder()
ohe.fit([[1],[2],[3],[4]])
print(ohe.transform([[2],[3],[1],[4]]).toarray())

ohe = OneHotEncoder()
ohe.fit([[10],[3],[2],[1]])
print(ohe.transform([[2],[3],[1],[4],[6]]).toarray())

ohe = OneHotEncoder()
ohe.fit([[1,0],[2,1],[3,0],[4,1]])
print(ohe.transform([[2,0],[3,0],[1,1],[4,1]]).toarray())

# label
le = LabelEncoder()
le.fit(['男','女','未知'])
print(le.transform(['男','未知','女']))
print(le.transform(['男','女','未知']))

print([[1],[2],[3],[4]])
