import json
b = '{"code":"0","data":{"block":[{"type":"text","line":[{"confidence":1,"location":{"top_left":{"x":240,"y":284},"right_bottom":{"x":414,"y":308}},"word":[{"location":{"top_left":{"x":240,"y":284},"right_bottom":{"x":413,"y":308}},"content":"-产品名称：文胸"}]},{"confidence":1,"location":{"top_left":{"x":240,"y":308},"right_bottom":{"x":526,"y":337}},"word":[{"location":{"top_left":{"x":240,"y":308},"right_bottom":{"x":524,"y":337}},"content":"品执行标准：Q/CYAMN0001-2013"}]},{"confidence":1,"location":{"top_left":{"x":0,"y":331},"right_bottom":{"x":414,"y":372}},"word":[{"location":{"top_left":{"x":0,"y":331},"right_bottom":{"x":412,"y":372}},"content":"LC11Q11-P3A-A80呈合等级：合格品"}]},{"confidence":1,"location":{"top_left":{"x":63,"y":373},"right_bottom":{"x":158,"y":398}},"word":[{"location":{"top_left":{"x":63,"y":373},"right_bottom":{"x":156,"y":398}},"content":"A80"}]},{"confidence":1,"location":{"top_left":{"x":16,"y":395},"right_bottom":{"x":190,"y":416}},"word":[{"location":{"top_left":{"x":16,"y":395},"right_bottom":{"x":80,"y":416}},"content":"PRICE"},{"location":{"top_left":{"x":80,"y":395},"right_bottom":{"x":188,"y":416}},"content":"价格：1480.00"}]},{"confidence":1,"location":{"top_left":{"x":207,"y":360},"right_bottom":{"x":413,"y":410}},"word":[{"location":{"top_left":{"x":207,"y":360},"right_bottom":{"x":411,"y":410}},"content":"罗携批号：01"}]},{"confidence":1,"location":{"top_left":{"x":256,"y":389},"right_bottom":{"x":398,"y":425}},"word":[{"location":{"top_left":{"x":256,"y":389},"right_bottom":{"x":396,"y":425}},"content":"“检验：1"}]}]}]},"desc":"success","sid":"wcr00027ad2@dx25d90eaa40e46f2300"}'

splitors = ['：', ':']
dict = {}

with open('kv.dict', encoding='utf-8') as f:
    for line in f:
        words = line.strip().split(' ')
        dict[words[0]] = words

def format(str):
    """
    格式化json，并提取其中有效的内容
    :param str:
    :return:
    """
    obj = json.loads(str)
    content = set()

    for x in obj['data']['block'][0]['line']:
        for y in x['word']:
            # 过滤长度小于2的内容
            if(len(y['content'])>2):
                content.add(y['content'])

    return content

def iskvStyle(s):
    """
    分割键值对
    :param s:
    :return:
    """
    for splitor in splitors:
        if s.find(splitor) != -1:
            return s.split(splitor)
    return []

def getDict(a):
    """
    查找字典里面的key
    :param a:
    :return:
    """
    for k, v in dict.items():
        if a in v:
            return k
    return ''

def parseContent(arr):
    """
    提取关键信息
    :param arr:
    :return:
    """
    for i,v in enumerate(arr):
        if i != len(arr)-1:
            key = getDict(v)
            if key != '':
                return (key,arr[i+1])
    None

def parse(json):
    result = []

    for line in format(json):
        arr = iskvStyle(line)
        if len(arr) > 0:
            parsed = parseContent(arr)
            if parsed != None:
                result.append(parsed)

    return result

parse(b)
