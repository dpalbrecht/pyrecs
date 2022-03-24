import pytest
import numpy as np
from pyrecs.predict import tfrs_streaming


@pytest.mark.parametrize("inputs_dict,expected_predictions",
                         [
                             ({'user_identifiers':np.array(['user1', 'user2']), 
                               'user_embeddings':np.array([[1,2],[1,5]]), 
                               'item_identifiers':np.array(['a','b','c']), 
                               'item_embeddings':np.array([[1,2],[1,2],[1,2]]),
                               'user2excludeitems':{},
                               'n_recs':10},
                              {'user1': ['a', 'b', 'c'], 'user2': ['a', 'b', 'c']}),
                             ({'user_identifiers':np.array(['user1', 'user2']), 
                               'user_embeddings':np.array([[0,1],[1,0]]), 
                               'item_identifiers':np.array(['a','b','c']), 
                               'item_embeddings':np.array([[0,1],[0,0],[1,0]]),
                               'user2excludeitems':{},
                               'n_recs':1},
                              {'user1': ['a'], 'user2': ['c']}),
                             ({'user_identifiers':np.array(['user1', 'user2']), 
                               'user_embeddings':np.array([[1,2],[1,5]]), 
                               'item_identifiers':np.array(['a','b','c']), 
                               'item_embeddings':np.array([[1,2],[1,2],[1,2]]),
                               'user2excludeitems':{'user1':['b']},
                               'n_recs':2},
                              {'user1': ['a', 'c'], 'user2': ['a', 'b']}),
                             ({'user_identifiers':np.array(['user1', 'user2']), 
                               'user_embeddings':np.array([[1,0,0],[0,0,1]]), 
                               'item_identifiers':np.array(['a','b','c']), 
                               'item_embeddings':np.array([[1,0,0],[0,0,0],[0,0,1]]),
                               'user2excludeitems':{'user1':['c','a']},
                               'n_recs':1},
                              {'user1': ['b'], 'user2': ['c']})
                         ]
                        )
def test_predictions(inputs_dict, expected_predictions):
    predictions = tfrs_streaming.predict(user_identifiers=inputs_dict['user_identifiers'], 
                                         user_embeddings=inputs_dict['user_embeddings'],
                                         item_identifiers=inputs_dict['item_identifiers'], 
                                         item_embeddings=inputs_dict['item_embeddings'],
                                         n_recs=inputs_dict['n_recs'],
                                         user2excludeitems=inputs_dict['user2excludeitems'])
    assert predictions == expected_predictions
    

# @pytest.mark.parametrize("inputs_dict,expected_predictions",
#                          [
#                              ({'user_identifiers':np.array(['user1', 'user2']), 
#                                'user_embeddings':np.array([[1,2],[1,5]]), 
#                                'item_identifiers':np.array(['a','b','c']), 
#                                'item_embeddings':np.array([[1,2],[1,2],[1,2]]),
#                                'train_user_interactions':{},
#                                'popular_items':[],
#                                'n_recs':10},
#                               {'user1': ['a', 'b', 'c'], 'user2': ['a', 'b', 'c']})
#                          ]
#                         )
# def test_predictions2(inputs_dict, expected_predictions):
#     predictions = tfrs_streaming.predict2(user_identifiers=inputs_dict['user_identifiers'], 
#                                          user_embeddings=inputs_dict['user_embeddings'],
#                                          item_identifiers=inputs_dict['item_identifiers'], 
#                                          item_embeddings=inputs_dict['item_embeddings'],
#                                          n_recs=inputs_dict['n_recs'],
#                                          train_user_interactions=inputs_dict['train_user_interactions'],
#                                          popular_items=inputs_dict['popular_items'])
#     assert predictions == expected_predictions