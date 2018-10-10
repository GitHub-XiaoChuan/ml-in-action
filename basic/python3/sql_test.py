# -- coding: utf-8 --
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
import time
# 数据源
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

class Image_ocr(Base):
    # 表名
    __tablename__ = 'image_ocr'

    # 结构
    id = sa.Column(sa.BIGINT, primary_key=True)
    order_id = sa.Column(sa.BIGINT)
    order_item_id = sa.Column(sa.BIGINT, unique=True)
    item_id = sa.Column(sa.BIGINT)
    category_four_id = sa.Column(sa.BIGINT)
    category_three_id = sa.Column(sa.BIGINT)
    category_two_id = sa.Column(sa.BIGINT)
    category_one_id = sa.Column(sa.BIGINT)
    category_one_id_name = sa.Column(sa.VARCHAR(200))
    brand_id = sa.Column(sa.BIGINT)
    brand_name = sa.Column(sa.VARCHAR(200))
    store_id = sa.Column(sa.BIGINT)
    store_name = sa.Column(sa.VARCHAR(200))
    area_id = sa.Column(sa.BIGINT)
    area_name = sa.Column(sa.VARCHAR(200))
    sold_type = sa.Column(sa.BIGINT)
    sold_type_name = sa.Column(sa.VARCHAR(200))
    sold_area = sa.Column(sa.BIGINT)
    sold_area_name = sa.Column(sa.VARCHAR(200))
    counter_id = sa.Column(sa.BIGINT)
    counter_name = sa.Column(sa.VARCHAR(200))
    name = sa.Column(sa.VARCHAR(200))
    url = sa.Column(sa.VARCHAR(200))
    rec_context = sa.Column(sa.TEXT)
    modify_context = sa.Column(sa.TEXT)
    parse_context = sa.Column(sa.TEXT)
    hh = sa.Column(sa.VARCHAR(100))
    type = sa.Column(sa.INT)
    source = sa.Column(sa.INT)
    order_ship_time = sa.Column(sa.TIMESTAMP)
    modify_time = sa.Column(sa.TIMESTAMP)

img = session.query(Image_ocr).filter_by(id=1).first()
print(img.order_ship_time)
#print(img.modify_time)

#img.modify_time = datetime.now()
img.order_ship_time = datetime.now()

time.sleep(3)

print(img.order_ship_time)
#print(img.modify_time)

session.commit()

time.sleep(3)

print(img.order_ship_time)
#print(img.modify_time)
