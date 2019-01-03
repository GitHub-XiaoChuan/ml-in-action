"""
http://www.shareditor.com/blogshow?blogId=77
12 教你如何利用强大的中文语言技术平台做依存句法和语义依存分析

语义依存 和 语法依存

依存关系的公理：

1. 一个句子中只有一个成分是独立的
2. 其他成分直接依存于某一成分
3. 任何一个成分都不能依存于两个或两个以上的成分
4. 如果A成分直接依存于B成分，而C成分在句子中位于A和B之间，那么C或者直接依存于B，或者直接依存于A和B之间的某一成分
5. 中心成分左右两面的其他成分相互不发生关系

LTP依存关系标记

主谓关系        SBV subject-verb                我送她一束花 (我 <-- 送)
动宾关系        VOB 直接宾语，verb-object        我送她一束花 (送 --> 花)
间宾关系        IOB 间接宾语，indirect-object    我送她一束花 (送 --> 她)
前置宾语        FOB 前置宾语，fronting-object    他什么书都读 (书 <-- 读)
兼语           DBL double                       他请我吃饭 (请 --> 我)
定中关系        ATT attribute                   红苹果 (红 <-- 苹果)
状中结构        ADV adverbial                   非常美丽 (非常 <-- 美丽)
动补结构        CMP complement                  做完了作业 (做 --> 完)
并列关系        COO coordinate                  大山和大海 (大山 --> 大海)
介宾关系        POB preposition-object          在贸易区内 (在 --> 内)
左附加关系      LAD left adjunct                 大山和大海 (和 <-- 大海)
右附加关系      RAD right adjunct                孩子们 (孩子 --> 们)
独立结构        IS independent structure        两个单句在结构上彼此独立
核心关系        HED head                        指整个句子的核心

"""