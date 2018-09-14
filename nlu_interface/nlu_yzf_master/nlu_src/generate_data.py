# -*- coding : utf-8 -*-
"""
Created on Sat Mar 4 2017
@author : Aiting Liu
"""
import os

RAWDIR = '../nlu_raw/'
OUTPUT = '../nlu_data/'
if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)
filelist = ['train_normal', 'valid_normal', 'test_normal']
SLOT_LIST = ['S_OPE', 'S_ATTR', 'S_WAY', 'S_LOC', 'S_NAME']


def process(raw_file):
    f = open(os.path.join(RAWDIR, raw_file), 'r')
    lines = f.readlines()
    f.close()
    sents = []
    slots1, slots2, slots3, slots4, slots5 = [], [], [], [], []
    intents = []
    for i in range(0, len(lines)):
        line = lines[i].strip()
        tmp = line.split('\t')
        print(tmp)
        sent_tmp = ' '.join(list(tmp[0]))
        print(sent_tmp)
        S_OPE_tmp, S_ATTR_tmp, S_WAY_tmp, S_LOC_tmp, S_NAME_tmp = \
            tmp[2], tmp[3], tmp[4], tmp[5], tmp[6]
        INTENT_tmp = tmp[1]
        print(S_OPE_tmp, S_ATTR_tmp, S_WAY_tmp, S_LOC_tmp, S_NAME_tmp, INTENT_tmp)
        sents.append(' ' + sent_tmp + ' ' + '\n')
        slots1.append(' ' + S_OPE_tmp + ' ' + '\n')
        slots2.append(' ' + S_ATTR_tmp + ' ' + '\n')
        slots3.append(' ' + S_WAY_tmp + ' ' + '\n')
        slots4.append(' ' + S_LOC_tmp + ' ' + '\n')
        slots5.append(' ' + S_NAME_tmp + ' ' + '\n')
        intents.append(' ' + INTENT_tmp + ' ' + '\n')

    OUTPUTDIR = ''

    if 'train' in raw_file:
        OUTPUTDIR = os.path.join(OUTPUT, 'train')
        if not os.path.exists(OUTPUTDIR):
            os.makedirs(OUTPUTDIR)
    elif 'test' in raw_file:
        OUTPUTDIR = os.path.join(OUTPUT, 'test')
        if not os.path.exists(OUTPUTDIR):
            os.makedirs(OUTPUTDIR)
    elif 'valid' in raw_file:
        OUTPUTDIR = os.path.join(OUTPUT, 'valid')
        if not os.path.exists(OUTPUTDIR):
            os.makedirs(OUTPUTDIR)

    f = open(os.path.join(OUTPUTDIR, raw_file + '_sent.txt'), 'w')
    f.writelines(sents)
    f.close()
    f = open(os.path.join(OUTPUTDIR, raw_file + '_s_ope.txt'), 'w')
    f.writelines(slots1)
    f.close()
    f = open(os.path.join(OUTPUTDIR, raw_file + '_s_attr.txt'), 'w')
    f.writelines(slots2)
    f.close()
    f = open(os.path.join(OUTPUTDIR, raw_file + '_s_way.txt'), 'w')
    f.writelines(slots3)
    f.close()
    f = open(os.path.join(OUTPUTDIR, raw_file + '_s_loc.txt'), 'w')
    f.writelines(slots4)
    f.close()
    f = open(os.path.join(OUTPUTDIR, raw_file + '_s_name.txt'), 'w')
    f.writelines(slots5)
    f.close()
    f = open(os.path.join(OUTPUTDIR, raw_file + '_intent.txt'), 'w')
    f.writelines(intents)
    f.close()


def main():
    for r_file in filelist:
        process(r_file)
    print('Done.')

if __name__ == '__main__':
    main()
