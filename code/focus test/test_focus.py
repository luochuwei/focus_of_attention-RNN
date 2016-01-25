#-*- coding:utf-8 -*-
#############################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 28/12/2015
#    Usage: try to get sentence focus by extracting keywords
#
#############################################################

import jieba
import jieba.analyse
import numpy as np

def open_file(post_path, response_path):
    print "load original data"
    post = []
    f1 = open(post_path)
    for i in f1:
        post.append(i[:-1].split('\t'))
    f1.close()

    f2 = open(response_path)
    response = []
    for j in f2:
        response.append(j[:-1].split('\t'))
    f2.close()
    print "finishing loading post and response"

    assert len(post) == len(response)

    p_r = {}

    for i in range(len(post)):
        assert len(post[i]) == 2
        if post[i][-1] not in p_r:
            p_r[post[i][-1]] = [response[i][-1]]
        else:
            p_r[post[i][-1]].append(response[i][-1])

    print "get post response dict"
    pid_p_r = {}

    n = 0
    for i,j in p_r.iteritems():
        pid_p_r[n] = [i,j]
        n+=1
    return pid_p_r

def print_post_response(num_of_post, pid_p_r):
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "post : ", pid_p_r[num_of_post][0]
    print "response is below"
    for i in pid_p_r[num_of_post][1]:
        print i.decode('utf-8')
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
 
def get_focus(num_of_post, pid_p_r):
    s = pid_p_r[num_of_post][0]
    for i in pid_p_r[num_of_post][1]:
        s += i
    tfidf_list = jieba.analyse.extract_tags(s, allowPOS = ('ns','n', 'vn', 'v'), withWeight = True)
    text_rank_list = jieba.analyse.textrank(s, allowPOS = ('ns','n', 'vn', 'v'), withWeight = True)
    focus_dic = {}
    for (i,j) in tfidf_list:
        focus_dic[i] = j
    for (i,j) in text_rank_list:
        if i in focus_dic:
            focus_dic[i] += j
        else:
            focus_dic[i] = j
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print num_of_post," post : ", pid_p_r[num_of_post][0]
    print "response : "
    #print focus in response
    for i in pid_p_r[num_of_post][1]:
        word = (' '.join(jieba.cut(i))).split(' ')
        focus_c = []
        for j in word:
            if j in focus_dic:
                focus_c.append((j, focus_dic[j]))
        focus_c = sorted(focus_c, key = lambda x:x[-1], reverse = True)
        if focus_c != []:
            print i.decode('utf-8'),"--> focus is ",focus_c[0][0]
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

            


#main
post_path = r'repos-id-post-cn'
response_path = r'repos-id-cmnt-cn'
pid_p_r = open_file(post_path, response_path)

test_list = np.random.random_integers(0, len(pid_p_r), 10)

num_of_post = test_list[np.random.randint(len(test_list))]

get_focus(num_of_post, pid_p_r)
