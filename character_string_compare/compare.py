#-*= coding: utf-8 -*-
'''
@FileName : compare.py
@Time     : 2020/7/20 17:33:12
改自作者bojone的程序
'''
#! -*- coding: utf-8 -*-

from A_snippets import longest_common_subsequence


def red_color(text):
    return u'\033[1;31;40m%s\033[0m' % text


def blue_color(text):
    return u'\033[1;32;40m%s\033[0m' % text


def String_compare(origin, objective):
    _, mappings = longest_common_subsequence(origin, objective)
    source_idxs = set([i for i, j in mappings])
    target_idxs = set([j for i, j in mappings])
    colored_origin, colored_target = u'', u''
    for i, j in enumerate(origin):
        if i in origin_idxs:
            colored_origin += blue_color(j)
        else:
            colored_origin += red_color(j)
    for i, j in enumerate(objective):
        if i in objective_idxs:
            colored_objective += blue_color(j)
        else:
            colored_objective += red_color(j)
    print(colored_origin)
    print(colored_objective)
