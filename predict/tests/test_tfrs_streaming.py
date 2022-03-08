import pytest
import numpy as np
from pyrecs.predict import tfrs_streaming


@pytest.mark.parametrize("inputs_dict,expected_predictions",
                         [
                             ({'user_identifiers':np.array(['user1', 'user2']), 
                               'user_embeddings':np.array([[1,2],[1,5]]), 
                               'item_identifiers':np.array(['a','b','c']), 
                               'item_embeddings':np.array([[1,2],[1,2],[1,2]])},
                              {'user1': ['a', 'b', 'c'], 'user2': ['a', 'b', 'c']})
                         ]
                        )
def test_predictions(inputs_dict, expected_predictions):
    predictions = tfrs_streaming.predict(user_identifiers=inputs_dict['user_identifiers'], 
                                         user_embeddings=inputs_dict['user_embeddings'],
                                         item_identifiers=inputs_dict['item_identifiers'], 
                                         item_embeddings=inputs_dict['item_embeddings'],
                                         embedding_dtype='float32', 
                                         n_recs=10,
                                         prediction_batch_size=32)
    assert predictions == expected_predictions