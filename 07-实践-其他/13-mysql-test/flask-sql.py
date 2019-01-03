from sqlalchemy import create_engine

# 参考：https://www.cnblogs.com/zhangxinqi/p/8480049.html
engine = create_engine('mysql+pymysql://user:passwd@127.0.0.1:3306/test')
result = engine.execute('select * from standby')
engine.execute("insert into test1(id,name,salary) values(1,'zs',88888)")
print(result.fetchall())