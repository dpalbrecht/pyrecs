import tensorflow_recommenders as tfrs
import tensorflow as tf


def predict(user_identifiers, user_embeddings,
            item_identifiers, item_embeddings,
            train_users, test_users,
            embedding_dtype='float32', 
            n_recs=10,
            prediction_batch_size=32):
    
    # Assumptions check
    try:
        assert len(set(user_identifiers) & set(train_users+test_users)) == len(set(user_identifiers))
    except:
        raise AssertionError('All user identifiers must be in one or both of train/test users.')
    train_users = {t:1 for t in train_users}
    test_users = {t:1 for t in test_users}
    
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
    predictions = {'train':{},'test':{}}
    for user_ids, user_embs in user_chunks:
        _, candidate_ids_list = stream(user_embs, k=n_recs)
        candidate_ids_list = candidate_ids_list.numpy().astype(str).tolist()
        for user_id, candidate_ids in zip(user_ids, candidate_ids_list):
            if train_users.get(user_id) is not None:
                predictions['train'][user_id] = candidate_ids
            if test_users.get(user_id) is not None:
                predictions['test'][user_id] = candidate_ids
    
    return predictions