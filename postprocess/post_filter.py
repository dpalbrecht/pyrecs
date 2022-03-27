def post_filter(predictions,
                n_recs,
                popular_items=[],
                train_user2interactions={},
                test_user2interactions={}):
    
    # For users in test with train interactions, we want to:
    # 1) Filter out those interactions if provided
    # 2) Fill them with popular items unless we have enough recommendations already (set n_recs in predict higher than here)
    for user_id, test_candidates in predictions['test'].items():
        train_interactions = train_user2interactions.get(user_id)
        if train_interactions is not None:
            test_candidates = [c for c in test_candidates if (c not in train_interactions)]
            if (len(test_candidates) < n_recs) & (len(popular_items) > 0):
                for pop_item in popular_items:
                    if (pop_item not in train_interactions) & (pop_item not in test_candidates):
                        test_candidates.append(pop_item)
                    if len(test_candidates) == n_recs:
                        break
        predictions['test'][user_id] = test_candidates[:n_recs]
        
    # For users in train, we want to make sure we have the right number of recommendations
    for user_id, train_candidates in predictions['train'].items():
        predictions['train'][user_id] = train_candidates[:n_recs]
        
    # For users in test_user2interactions that do not have predictions, make predictions
    for user_id in test_user2interactions.keys():
        if predictions['test'].get(user_id) is None:
            predictions['test'][user_id] = popular_items[:n_recs]
    
    return predictions