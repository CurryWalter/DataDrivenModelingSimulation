import covasim as cv
import datetime as dt
import numpy as np

SCHOOL = 's'
WORK   = 'w'
# the original edge clipping intervention does not allow for age-based edge clipping,
# this intervention takes in a age ranges and pops edges for those age groups on specified weekdays and layers
class age_based_edge_clipping(cv.Intervention):
    
    def __init__(self, ranges=[],
                 start_day=None, end_day=None,
                  **kwargs):
        super().__init__(**kwargs)
       
        self.start_day           = start_day
        self.end_day             = end_day
        self.age_ranges          = ranges
        self.popped_edges        = {layer: [] for layer in (SCHOOL, WORK)}
        self.is_popped           = {layer: False for layer in (SCHOOL, WORK)}


    def initialize(self, sim):
        super().initialize()

        self.start_ti = sim.day(self.start_day)
        self.end_ti   = sim.day(self.end_day)

        start_day_raw = sim['start_day']

        # convert start_day_raw to a datetime object for weekday calculations
        self._start_date = dt.datetime.strptime(str(start_day_raw), '%Y-%m-%d')
        
        self.home_by_day_by_age = {age_range['age']: np.zeros(sim.npts, dtype=bool) for age_range in self.age_ranges}

        for t in range(sim.npts):
            if self.start_ti <= t <= self.end_ti:
                weekday_t = (self._start_date + dt.timedelta(days=t)).weekday()
                for age_range in self.age_ranges:
                    self.home_by_day_by_age[age_range['age']][t] = weekday_t in age_range['home_weekdays']
                

    def apply(self, sim):
        for age_range in self.age_ranges:
            self._clip(sim, self.home_by_day_by_age[age_range['age']], age_range['age'], layer=age_range['layer'])

    #similar to the edge clipping intervention step function but instead of maintaining an original edge list,
    #we maintain a list of popped edges for each layer and restore them when the intervention is over
    # unlike the original edge clipping function, we do not need to do the math for which edges to keep
    def _clip(self, sim, home_by_day, age_range, layer):
        t = sim.ti

        should_be_home = home_by_day[t]

        if should_be_home and not self.is_popped[layer]:
            self._pop_age_group(sim, age_range, layer)
            self.is_popped[layer] = True
        elif not should_be_home and self.is_popped[layer]:
            self._restore_edges(sim, layer)
            self.is_popped[layer] = False

    def _pop_age_group(self, sim, age_range, layer):
        network_layer   = sim.people.contacts[layer]
        person1, person2 = network_layer['p1'], network_layer['p2']
        ages  = sim.people.age
        alive = ~sim.people.dead          

        age_mask = ((ages[person1] >= age_range[0]) & (ages[person1] <= age_range[1])) | \
                ((ages[person2] >= age_range[0]) & (ages[person2] <= age_range[1]))

        alive_mask = alive[person1] & alive[person2]

        mask = age_mask & alive_mask
        target_indices = np.where(mask)[0]

        self.popped_edges[layer].append(network_layer.pop_inds(target_indices))


    def _restore_edges(self, sim, layer):
        network_layer   = sim.people.contacts[layer]
        alive = ~sim.people.dead 
        while self.popped_edges[layer]:
            popped = self.popped_edges[layer].pop()

            keep = alive[popped['p1']] & alive[popped['p2']]    
            if keep.any():
                filtered = {k: v[keep] for k, v in popped.items()}
                network_layer.append(filtered)

#to avoid memory issues and warning logs, we clear the popped edges after the intervention is over
    def shrink(self):
        self.popped_edges = {layer: [] for layer in (SCHOOL, WORK)}
        return

strategy = [{'age':(5,10),'home_weekdays': {0,2,4}, 'layer': SCHOOL},
              {'age':(11,17),'home_weekdays': {1,3,4}, 'layer': SCHOOL},
               {'age':(18,30),'home_weekdays': {0,2,4}, 'layer': WORK},
              {'age':(31,55),'home_weekdays': {1,3}, 'layer': WORK},
              ]

AGE_BASED_SOCIAL_DISTANCING = age_based_edge_clipping(
   ranges=strategy,
    start_day='2020-03-01',             
    end_day='2021-03-01',
)

VACCINATION = cv.simple_vaccine(days=[150, 171], prob=0.75, rel_sus=0.5, rel_symp=0.5)

PREMATURE_OPENING = [
    cv.change_beta(days=[15, 60], changes=[0.5, 1.0], layers='s'),
    cv.change_beta(days=[15, 60], changes=[0.5, 1.0], layers='w'),
    cv.change_beta(days=[15, 60], changes=[0.6, 1.0], layers='c'),
    cv.test_prob(
        start_day=15,
        symp_prob=0.15,
        asymp_prob=0.01,
        symp_quar_prob=0.8,
        asymp_quar_prob=0.5,
        test_delay=1,
    ),
    cv.contact_tracing(
        start_day=15,
        trace_probs={'h': 1.0, 's': 0.5, 'w': 0.4, 'c': 0.2},
        trace_time=1,
    ),
    ]    

def testing(start):
        return cv.test_prob(symp_prob=0.2, asymp_prob=0.001,
                            symp_quar_prob=0.99, start_day=start, test_delay=1)

def tracing(start):
    return cv.contact_tracing(trace_probs=dict(h=1.0, s=0.5, w=0.5, c=0.2),
                                trace_time=dict(h=0, s=2, w=2, c=5),
                                start_day=start)


def get_scenarios():
    """Define scenarios"""
    baseline = []
    scenario_1 = AGE_BASED_SOCIAL_DISTANCING
    scenario_2 = []
    scenario_3 = PREMATURE_OPENING
    scenario_4 = [testing(20), tracing(20)]
    scenario_5 = [VACCINATION]
    
    return {
        'Age Based Social Distancing': scenario_1,
        'Scenario 2': scenario_2,
        'Premature Reopening': scenario_3,
        'testing & tracing from day 20': scenario_4,
        'Vaccination from day 150': scenario_5,
    }
