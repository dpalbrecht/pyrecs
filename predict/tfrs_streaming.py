import tensorflow_recommenders  as tfrs
import tensorflow as tf
import itertools


def predict(user_identifiers, user_embeddings,
            item_identifiers, item_embeddings,
            embedding_dtype='float32', 
            n_recs=10,
            prediction_batch_size=32):
    
    # Create tfrs.Streaming instance
    stream = tfrs.layers.factorized_top_k.Streaming()
    stream.index_from_dataset(
        tf.data.Dataset.from_tensor_slices({
            'identifier':item_identifiers,
            'embedding':item_embeddings.astype(embedding_dtype)
        }).batch(1).map(lambda x: (x['identifier'], x['embedding']))
    )
    
    # Get batched predictions
    if len(user_embeddings.shape) == 1:
        user_embeddings = user_embeddings[None,:]
    input_embeddings = tf.constant(user_embeddings, dtype=embedding_dtype)
    predictions = tf.data.Dataset.from_tensor_slices(input_embeddings)\
                    .batch(prediction_batch_size)\
                    .map(lambda x: stream(x, k=n_recs)[1])
    
    # Format predictions
    formatted_predictions = []
    for n, candidate_ids in enumerate(itertools.chain(*predictions.as_numpy_iterator())):
        formatted_predictions.append((user_identifiers[n], 
                                      candidate_ids.flatten().astype(str).tolist()))
        
    return dict(formatted_predictions)