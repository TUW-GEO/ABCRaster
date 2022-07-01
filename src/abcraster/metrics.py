import math


def overall_accuracy(conf):
    return (conf['TP'] + conf['TN']) / (conf['TP'] + conf['TN'] + conf['FP'] + conf['FN'])


def kappa(conf):
    pe1 = (conf['TP'] + conf['FN']) * (conf['TP'] + conf['FP']) + (conf['FP'] + conf['TN']) * (conf['FN'] + conf['TN'])
    pe2 = conf['TP'] + conf['TN'] + conf['FP'] + conf['FN']
    pe = pe1 / pe2 ** 2
    po = overall_accuracy(conf)
    return (po - pe) / (1 - pe)


def users_producers_accuracy(conf):
    ua = conf['TP'] / (conf['TP'] + conf['FP'])
    pa = conf['TP'] / (conf['TP'] + conf['FN'])  # accuracy:PP2 as defined in ACube4Floods 5.1
    return ua, pa


def critical_success_index(conf):
    return conf['TP'] / (conf['TP'] + conf['FP'] + conf['FN'])


def f1_score(conf):
    return (2 * conf['TP']) / (2 * conf['TP'] + conf['FN'] + conf['FP'])


def comission_omission_error(conf):
    ce = conf['FP'] / (conf['FP'] + conf['TP'])  # inverse of precision
    oe = conf['FN'] / (conf['FN'] + conf['TP'])  # inverse of recall
    return ce, oe


def penalization(conf):
    # penalization as defined in ACube4Floods 5.1
    return math.exp(conf['FP'] / ((conf['TP'] + conf['FN']) / math.log(0.5)))


def success_rate(conf):
    # Success rate as defined in ACube4Floods 5.1
    ua, pa = users_producers_accuracy(conf)
    p = penalization(conf)
    return pa - (1 - p)


def bias(conf):
    # bias as shown in the GFM proposal
    return (conf['TP'] + conf['FP']) / (conf['TP'] + conf['FN'])


def prevalence(conf):
    # prevalence as defined by Dasgupta (in preparation)
    return (conf['TP'] + conf['FN']) / (conf['TP'] + conf['FN'] + conf['TN'] + conf['FP'])
