import pytest
from pyrecs.postprocess import post_filter


@pytest.mark.parametrize("inputs_dict,expected_predictions",
                         [
                             ({'predictions':{'train':{'user1':['a','b','c']},
                                              'test':{'user1':['a','b','c']}}, 
                               'n_recs':3, 
                               'popular_items':[], 
                               'train_user2interactions':{'user1':['a','b']},
                               'test_user2interactions':{}},
                              {'train':{'user1': ['a','b','c']}, 
                               'test':{'user1': ['c']}}),
                             ({'predictions':{'train':{'user1':['a','b','c']},
                                              'test':{'user1':['a','b','c']}}, 
                               'n_recs':3, 
                               'popular_items':['d'], 
                               'train_user2interactions':{'user1':['a','b']},
                               'test_user2interactions':{}},
                              {'train':{'user1': ['a','b','c']}, 
                               'test':{'user1': ['c','d']}})
                         ]
                        )
def test_1_user(inputs_dict, expected_predictions):
    predictions = post_filter.post_filter(**inputs_dict)
    assert predictions == expected_predictions
    
    
@pytest.mark.parametrize("inputs_dict,expected_predictions",
                         [
                             ({'predictions':{'train':{'user1':['a','b','c'], 
                                                       'user2':['b','c','d']},
                                              'test':{'user1':['a','b','c']}}, 
                               'n_recs':3, 
                               'popular_items':['d','e'], 
                               'train_user2interactions':{'user1':['a','b']},
                               'test_user2interactions':{}},
                              {'train':{'user1': ['a','b','c'],
                                        'user2':['b','c','d']}, 
                               'test':{'user1': ['c','d','e']}})
                         ]
                        )
def test_multiple_user(inputs_dict, expected_predictions):
    predictions = post_filter.post_filter(**inputs_dict)
    assert predictions == expected_predictions
    

@pytest.mark.parametrize("inputs_dict,expected_predictions",
                         [
                             ({'predictions':{'train':{'user1':['a','b','c']},
                                              'test':{'user1':['a','b','c']}}, 
                               'n_recs':3, 
                               'popular_items':['a'], 
                               'train_user2interactions':{'user1':['a','b']},
                               'test_user2interactions':{'user2':['d']}},
                              {'train':{'user1': ['a','b','c']}, 
                               'test':{'user1': ['c'], 'user2':['a']}})
                         ]
                        )
def test_new_user(inputs_dict, expected_predictions):
    predictions = post_filter.post_filter(**inputs_dict)
    assert predictions == expected_predictions

    
@pytest.mark.parametrize("inputs_dict,expected_predictions",
                         [
                             ({'predictions':{'train':{'user1':['a','b','c','d','e']},
                                              'test':{'user2':['a','b','c','d','e']}}, 
                               'n_recs':3, 
                               'popular_items':[], 
                               'train_user2interactions':{},
                               'test_user2interactions':{}},
                              {'train':{'user1': ['a','b','c']}, 
                               'test':{'user2': ['a','b','c']}})
                         ]
                        )
def test_large_incoming_n_recs(inputs_dict, expected_predictions):
    predictions = post_filter.post_filter(**inputs_dict)
    assert predictions == expected_predictions