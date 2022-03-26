import tensorflow_recommenders  as tfrs
import tensorflow as tf
import itertools


# TODO: Add functionality to pass train_user_interactions={} and get predictions in train
        # Right now, I assume users not in train_user_interactions are test users and that's not true
        # So if I pass in train_user_interactions={} with the intent not to remove interactions, I end up with 0 train predictions
        # Maybe pass hash maps of train and test users, along with interactions to remove from train (if supplied)?
            # Then I can make recommendations for users without embeddings
def predict(user_identifiers, user_embeddings,
            item_identifiers, item_embeddings,
            train_user_interactions={},
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
    
    # Batch inputs
    if len(user_embeddings.shape) == 1:
        user_embeddings = user_embeddings[None,:]
    batch_ranges = range(0, len(user_embeddings), prediction_batch_size)
    user_chunks = ([user_identifiers[i:i + prediction_batch_size],
                    user_embeddings[i:i + prediction_batch_size].astype(embedding_dtype)] 
                   for i in batch_ranges)
    
    # Get predictions
    formatted_predictions = {'train':{}, 'test':{}}
    if len(train_user_interactions) > 0:
        max_exclude_len = max([len(v) for v in train_user_interactions.values()])
    else:
        max_exclude_len = 0
    for user_ids, user_embs in user_chunks:
        scores, candidate_ids_list = stream(user_embs, k=n_recs+max_exclude_len)
        candidate_ids_list = candidate_ids_list.numpy().astype(str)
        for user_id, candidate_ids in zip(user_ids, candidate_ids_list):
            candidate_ids = candidate_ids.tolist()
            
            train_interactions = train_user_interactions.get(user_id)
            
            # This is a user not in train. We need:
            # * Test recommendations where we don't remove interactions
            if train_interactions is None:
                formatted_predictions['test'][user_id] = candidate_ids[:n_recs]
            
            # This is a user in train. We need:
            # * Train recommendations where we don't remove interactions
            # * Test recommendations where we do remove interactions
            else:
                formatted_predictions['train'][user_id] = candidate_ids[:n_recs]
                train_candidates = []
                for cid in candidate_ids:
                    if cid not in train_interactions:
                        train_candidates.append(cid)
                    if len(train_candidates) == n_recs:
                        break
                formatted_predictions['test'][user_id] = train_candidates
        
    return formatted_predictions