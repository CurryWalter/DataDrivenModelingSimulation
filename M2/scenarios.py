import covasim as cv

VACCINATION = cv.simple_vaccine(days=[150, 171], prob=0.75, rel_sus=0.5, rel_symp=0.5)

def testing(start):
        return cv.test_prob(symp_prob=0.2, asymp_prob=0.001,
                            symp_quar_prob=0.99, start_day=start, test_delay=1)

def tracing(start):
    return cv.contact_tracing(trace_probs=dict(h=1.0, s=0.5, w=0.5, c=0.2),
                                trace_time=dict(h=0, s=2, w=2, c=5),
                                start_day=start)


def get_scenarios():
    """Define scenarios"""
    scenario_1 = []
    scenario_2 = []
    scenario_3 = []
    scenario_4 = [testing(20), tracing(20)]
    scenario_5 = [VACCINATION]
    
    return {
        'Scenario 1': scenario_1,
        'Scenario 2': scenario_2,
        'Scenario 3': scenario_3,
        'testing & tracing from day 20': scenario_4,
        'Vaccination from day 150': scenario_5,
    }
