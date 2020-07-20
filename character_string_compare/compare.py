#-*= coding: utf-8 -*-
'''
@FileName : compare.py
@Time     : 2020/7/20 17:33:12
'''
#! -*- coding: utf-8 -*-

from A_snippets import longest_common_subsequence


def red_color(text):
    return u'\033[1;31;40m%s\033[0m' % text


def blue_color(text):
    return u'\033[1;32;40m%s\033[0m' % text


def compare(source, target):
    _, mappingss = longest_common_subsequence(source, target)
    source_idxs = set([i for i, j in mappingss])
    target_idxs = set([j for i, j in mappingss])
    colored_source, colored_target = u'', u''
    for i, j in enumerate(source):
        if i in source_idxs:
            colored_source += blue_color(j)
        else:
            colored_source += red_color(j)
    for i, j in enumerate(target):
        if i in target_idxs:
            colored_target += blue_color(j)
        else:
            colored_target += red_color(j)
    print(colored_source)
    print(colored_target)