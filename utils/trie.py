# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from collections import defaultdict


class TreeNode():
    def __init__(self):
        self.child = defaultdict(TreeNode)

class Trie:

    def __init__(self, eos):
        self.root = TreeNode()
        self.eos = eos

    def insert(self, word):
        cur = self.root
        for c in word:
            cur = cur.child[c]

    # 综上所述，get_next_layer方法用于在字典树中找到以word为前缀的单词，并返回这些单词对应的节点索引列表。
    # 在上述代码中，self.eos用于表示结束标记（即EOS索引），它会被添加到返回的索引列表中，以表示没有匹配的下一层节点。
    def get_next_layer(self, word):
        cur = self.root
        for c in word:
            cur = cur.child.get(c)
            if cur is None:
                return [self.eos]
        return list(cur.child.keys())
