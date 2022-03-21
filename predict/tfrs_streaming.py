import tensorflow_recommenders  as tfrs
import tensorflow as tf
import itertools
    

def predict(user_identifiers, user_embeddings,
            item_identifiers, item_embeddings,
            user2excludeitems={},
            embedding_dtype='float32', 
            n_recs=10,
            prediction_batch_size=32):
    
    # Create tfrs.Streaming instance
    stream = tfrs.layers.factorized_top_k.Streaming()
    stream.index_from_dataset(
    tf.data.Dataset.from_tensors({
            'identifier':item_identifiers,
            'embedding':item_embeddings.astype(embedding_dtype)
        }).map(lambda x: (x['identifier'], x['embedding']))
    )
    
    # Get batched predictions
    if len(user_embeddings.shape) == 1:
        user_embeddings = user_embeddings[None,:]
    if len(user2excludeitems) == 0:
        pred_func = lambda x: (x['identifier'], stream(tf.squeeze(x['embedding']), k=n_recs)[1])
        exclusions = ['']*len(user_identifiers)
    else:
        pred_func = lambda x: (x['identifier'], 
                               stream.query_with_exclusions(tf.squeeze(x['embedding']), 
                                                            exclusions=x['exclusions'],
                                                            k=n_recs)[1])
        max_exclude_len = max([len(v) for v in user2excludeitems.values()])
        exclusions = []
        for user_id in user_identifiers:
            temp_vec = ['']*max_exclude_len
            for n, item in enumerate(user2excludeitems.get(user_id, [])):
                temp_vec[n] = item
            exclusions.append(temp_vec)
    predictions = tf.data.Dataset.from_tensor_slices({
                    'identifier':user_identifiers, 
                    'embedding':user_embeddings.astype(embedding_dtype),
                    'exclusions':exclusions
                })\
                .batch(prediction_batch_size)\
                .map(pred_func)
    
    # Format predictions
    formatted_predictions = {}
    for inputs, candidates in predictions:
        formatted_predictions.update(dict(zip(inputs.numpy().astype(str).tolist(), 
                                              candidates.numpy().astype(str).tolist())))
        
    return formatted_predictions