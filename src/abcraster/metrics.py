import math


def overall_accuracy(conf):
    """Overall Accuracy"""
    return (conf['TP'] + conf['TN']) / (conf['TP'] + conf['TN'] + conf['FP'] + conf['FN'])


def kappa(conf):
    """Kappa"""
    pe1 = (conf['TP'] + conf['FN']) * (conf['TP'] + conf['FP']) + (conf['FP'] + conf['TN']) * (conf['FN'] + conf['TN'])
    pe2 = conf['TP'] + conf['TN'] + conf['FP'] + conf['FN']
    pe = pe1 / pe2 ** 2
    po = overall_accuracy(conf)
    return (po - pe) / (1 - pe)


def users_accuracy(conf):
    """Users Accuracy"""
    ua = conf['TP'] / (conf['TP'] + conf['FP'])
    return ua


def producers_accuracy(conf):
    """Producers Accuracy"""
    pa = conf['TP'] / (conf['TP'] + conf['FN'])  # accuracy:PP2 as defined in ACube4Floods 5.1
    return pa


def critical_success_index(conf):
    """Critical Success Index"""
    return conf['TP'] / (conf['TP'] + conf['FP'] + conf['FN'])


def f1_score(conf):
    """F1 Score"""
    return (2 * conf['TP']) / (2 * conf['TP'] + conf['FN'] + conf['FP'])


def commission_error(conf):
    """commission error"""
    ce = conf['FP'] / (conf['FP'] + conf['TP'])  # inverse of precision
    return ce


def omission_error(conf):
    """omission error"""
    oe = conf['FN'] / (conf['FN'] + conf['TP'])  # inverse of recall
    return oe


def penalization(conf):
    """penalization"""
    # penalization as defined in ACube4Floods 5.1
    return math.exp(conf['FP'] / ((conf['TP'] + conf['FN']) / math.log(0.5)))


def success_rate(conf):
    """Success Rate"""
    # Success rate as defined in ACube4Floods 5.1
    pa = producers_accuracy(conf)
    p = penalization(conf)
    return pa - (1 - p)


def bias(conf):
    """Bias"""
    # bias as shown in the GFM proposal
    return (conf['TP'] + conf['FP']) / (conf['TP'] + conf['FN'])


def prevalence(conf):
    """Prevalence"""
    # prevalence as defined by Dasgupta (in preparation)
    return (conf['TP'] + conf['FN']) / (conf['TP'] + conf['FN'] + conf['TN'] + conf['FP'])


metrics = {'OA': overall_accuracy, 'K': kappa, 'CSI': critical_success_index, 'B': bias, 'P': prevalence,
           'UA': users_accuracy, 'PA': producers_accuracy, 'CE': commission_error, 'OE': omission_error,
           'SR': success_rate, 'F1': f1_score}

# metric dictionary is used to map metric 'keys' to functions
# docs strings to be used in output
