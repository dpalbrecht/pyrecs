import tensorflow_recommenders  as tfrs
import tensorflow as tf
import itertools
import copy
    

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


def predict2(user_identifiers, user_embeddings,
             item_identifiers, item_embeddings,
             train_user_interactions={},
             popular_items=[],
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
        
    batch_ranges = range(0, len(user_embeddings), prediction_batch_size)
    user_chunks = ([user_identifiers[i:i + prediction_batch_size],
                    user_embeddings[i:i + prediction_batch_size].astype(embedding_dtype)] 
                   for i in batch_ranges)
    
    train_predictions = {}
    max_exclude_len = max([len(v) for v in train_user_interactions.values()])
    for user_ids, user_embs in user_chunks:
        scores, candidate_ids_list = stream(user_embs, k=n_recs+max_exclude_len)
        for user_id, candidate_ids in zip(user_ids, candidate_ids_list):
            temp_popular_items = copy(popular_items)
            temp_exclusions = train_user_interactions.get(user_id)
            temp_candidates = []
            for cid in candidate_ids:
                if cid not in temp_exclusions:
                    temp_candidates.append(cid)
                if len(temp_candidates) == n_recs:
                    break
            if (len(temp_candidates) != n_recs) & (len(temp_popular_items) > 0): 
                for pop_item in temp_popular_items:
                    if (pop_item not in temp_exclusions) & (pop_item not in temp_candidates):
                        temp_candidates.append(pop_item)
                    if len(temp_candidates) == n_recs:
                        break
                
            train_predictions[user_id] = temp_candidates
        
    return formatted_predictions