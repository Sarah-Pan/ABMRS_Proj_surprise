from recommender import Recommender
from data import items, users, ratings
import pandas as pd

recommender = Recommender(ratings)

# Simulation loop
steps = 10
for step in range(steps):
    print(f"\n=== Simulation Step {step + 1} ===")
    
    users_due = users  # Optional: set the logic of choosing agent 
    
    for user_id in users_due['user_id']:
        print(f"\n[Agent] User {user_id} is active this round.")
        pred_scores = []
        for item_id_check in items['item_id']:
            score = recommender.predict_rating(user_id, item_id_check)
            pred_scores.append({
                'item_id': item_id_check,
                'item_name': items[items['item_id'] == item_id_check]['item_name'].values[0],
                'predicted_rating': round(score, 2)
                })
        
        pred_df = pd.DataFrame(pred_scores)
        print(f"\n  > Predicted scores for User {user_id}:")
        print(pred_df.to_string(index=False))  

        item_id = recommender.select_item(user_id)
        if item_id is None:
            print(f"  [SKIP] No items left to recommend for User {user_id}.")
            continue

        item_info = items[items['item_id'] == item_id].iloc[0]
        item_name = item_info['item_name']
        predicted_rating = recommender.predict_rating(user_id, item_id)
        print(f"  [RECOMMEND] Item {item_id} - '{item_name}' predicted rating: {predicted_rating:.2f}")

        rating = recommender.consume_item(user_id, item_id, predicted_rating)

        if rating is not None:
            print(f"  [CONSUME] User {user_id} consumed '{item_name}' with rating: {rating}")
        else:
            print(f"  [DECLINE] User {user_id} skipped '{item_name}' despite predicted rating: {predicted_rating:.2f}")

    # Optional: recompute populations after all users in the step
    recommender.update_populations()

# Print final item & user average ratings
recommender.print_populations()
