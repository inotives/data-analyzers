import pandas as pd
import random
from itertools import combinations
from collections import Counter

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

import plotly.express as px
import plotly.graph_objects as go

from utils.tools import load_csv_from_data

# Load and Preps the data ...
def load_and_pred_data():
    data = load_csv_from_data('sgtoto')
    
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    # Calculate the total sum of all winning numbers for each draw
    data['total_sum'] = data[['wnum_1', 'wnum_2', 'wnum_3', 'wnum_4', 'wnum_5', 'wnum_6']].sum(axis=1)

    # Ensure the index is sorted (in case it's not already)
    data = data.sort_index(ascending=True)

    
    return data

# Man Generator methods 
def win_and_huat(model='simple', num_set=1, plot_draw=False):
    data = load_and_pred_data()
    
    plot_all_draws(data) if plot_draw else None 
    print('Last Draw WINNING NUM::\n----------------------\n', data.iloc[-1])
    print(f"GENERATING NUMBER SETS::{num_set}")

    if model=='simple':
        for n in range(num_set):
            print(f"HUAT-AH::{weighted_random_pair_selection_generator(data)}")
    elif model=='fpg': 
        for n in range(num_set):
            print(f"HUAT-AH::{fp_growth_generator(data)}")
    elif model == 'afpg':
        for n in range(num_set):
            generated_numbers = adjusted_fp_growth_generator(data)
            print(f"\n>>>>>>>>> GENERATED NUM (HUAT-AH!!) SET({n+1}) ::  {generated_numbers}")
            print(f"Last 4 Draws Appearances::\n----------------------------------------")
            check_numbers_in_recent_draws(data, generated_numbers)


def check_numbers_in_recent_draws(data, generated_numbers, n=4):
    """
    Check the percentage of appearance of the generated numbers in the last 'n' draws.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing historical lottery data with date as index.
    - generated_numbers (list): List of generated numbers to check.
    - n (int): Number of recent draws to consider.

    Returns:
    - dict: A dictionary with the percentage of appearance for each generated number.
    """

    # Get the dates of the last 'n' draws
    recent_dates = data.index[-n:]

    # Filter the DataFrame for these recent draws
    recent_draws = data.loc[recent_dates]

    # Initialize a dictionary to store the counts
    counts = {num: 0 for num in generated_numbers}

    # Total number of recent draws
    total_draws = len(recent_draws)

    # Check each draw
    for _, row in recent_draws.iterrows():
        winning_numbers = set(row[['wnum_1', 'wnum_2', 'wnum_3', 'wnum_4', 'wnum_5', 'wnum_6']])
        for num in generated_numbers:
            if num in winning_numbers:
                counts[num] += 1

    # Calculate the percentage for each number
    percentages = {num: (count / total_draws) * 100 for num, count in counts.items()}

    # Combine counts and percentages into a single dictionary
    results = {num: {'count': counts[num], 'percentage': percentages[num]} for num in generated_numbers}
    print(recent_draws[['total_sum', 'odd_cnt','even_cnt', '1to10', '11to20', '21to30', '31to40', '41to49']])
    for k,r in results.items():
        print(f"NUM:{k} appear: {int(r['count'])}")

    return results


def analyse_repeated_number_occurance(data):
    # Step 1: Create a list to store results
    results = []

    # Step 2: Iterate through each draw
    for i, (date, row) in enumerate(data.iterrows()):
        # Get the winning numbers for the current draw
        current_draw_numbers = row.values
        
        # Calculate the previous 4 dates
        previous_dates = data.index[max(0, i+1):min(len(data), i+5)]
        
        # Initialize matching numbers set
        matching_numbers = set()
        
        # Check the previous draws
        if not previous_dates.empty:
            previous_draws = data.loc[previous_dates]
            
            # Find the actual numbers from the current draw that appeared in the last 4 draws
            for num in current_draw_numbers:
                if (previous_draws == num).any().any():
                    matching_numbers.add(num)
            
        # Count how many times the numbers from the current draw appeared in the last 4 draws
        count = len(matching_numbers)
        
        # Convert the matching numbers to a comma-separated string
        matching_numbers_str = ', '.join(map(str, sorted(matching_numbers)))
        
        # Append the result with the date, individual winning numbers, count, and the actual numbers
        results.append({
            'date': date,
            'wnum_1': row['wnum_1'],
            'wnum_2': row['wnum_2'],
            'wnum_3': row['wnum_3'],
            'wnum_4': row['wnum_4'],
            'wnum_5': row['wnum_5'],
            'wnum_6': row['wnum_6'],
            'appearances_in_last_4': count,
            'matching_numbers': matching_numbers_str
        })

    # Step 3: Convert results to a DataFrame for easy viewing
    results_df = pd.DataFrame(results).set_index('date')

    # Display the results
    print(results_df)


def plot_all_draws(data): 
    # Melt the DataFrame to have 'draw_num' and 'date' with corresponding 'winning_number'
    melted_data = data.reset_index().melt(id_vars=['date', 'draw_num', 'total_sum'], 
                                        value_vars=['wnum_1', 'wnum_2', 'wnum_3', 'wnum_4', 'wnum_5', 'wnum_6'], 
                                        var_name='number_position', value_name='winning_number')

    # Create a color mapping for different number positions
    color_map = {
        'wnum_1': 'blue',
        'wnum_2': 'green',
        'wnum_3': 'red',
        'wnum_4': 'orange',
        'wnum_5': 'purple',
        'wnum_6': 'cyan'
    }

    # Create a scatter plot for individual winning numbers
    scatter_fig = go.Scatter(
        x=melted_data['date'], 
        y=melted_data['winning_number'], 
        mode='markers', 
        marker=dict(color=melted_data['number_position'].map(color_map)),
        name='Winning Numbers',
        hovertext=melted_data['number_position']
    )

    # Create a line plot for the total sum of winning numbers
    line_fig = go.Scatter(
        x=data.index, 
        y=data['total_sum'], 
        mode='lines+markers', 
        name='Total Sum',
        line=dict(color='firebrick', width=2)
    )

    # Combine the scatter and line plots
    fig = go.Figure(data=[scatter_fig, line_fig])

    # Update layout for better readability
    fig.update_layout(
        title='Time Series of Lottery Draw Numbers and Total Sum',
        xaxis_title='Date',
        yaxis_title='Winning Numbers',
        legend_title='Legend',
        template='plotly_dark'
    )

    # Show the plot
    fig.show()



# Normal FP Growth Algo, but due to the randomness in lottery draw, it is always 1-49 get choosen to the final number pool hence it is still random number from 1-49 ...    
def fp_growth_generator(data): 
    ''' FP-GROWTH (Frequent Pattern) 
    a popular algorithm for mining frequent itemsets in large datasets, 
    which is particularly efficient compared to earlier methods like Apriori. 
    Commonly used in market basket analysis, where the goal is to find sets of items 
    (or "itemsets") that frequently appear together in transactions.

    Key Components
    ---------------
    1. Itemsets: a set of items that appear together in a transaction e.q. number set in lottery {1,14,34}
    2. Frequent itemset: one that appears in the dataset with a frequency above a certain threshold (known as minimum support).
    3. Support: a measure of how frequently an itemset appears in the dataset. 
       e.q if {1,14} appear in 1 out of 100 draws then support is 0.01 (1%)
    4. FP-Tree Construction: 
       The algorithm begins by constructing a Frequent Pattern Tree (FP-Tree). 
       This is a compressed representation of the dataset where each path represents 
       a set of items appearing together in transactions.
       Step to build FP tree: 
        a. Count Item Frequencies: Scan the dataset to count the frequency of each item.
        b. Order Items by Frequency: Only include items that meet the minimum support threshold 
           and order them by descending frequency.
        c. Tree Construction: For each transaction in the dataset, insert the items into the tree, 
           following the order established. If a path already exists for the items, 
           you just increment the count; otherwise, you create a new path.
    5. Mining the FP-Tree: 
       The next step is to mine the FP-Tree to find frequent itemsets.
       Conditional FP-Trees: For each item in the tree (starting from the least frequent), 
       you extract its "conditional FP-Tree," which represents all paths in the tree 
       that end with this item. You then recursively mine this smaller tree to find 
       all frequent itemsets that include the item. 
       The recursion continues until all items have been processed.
    6. Generating Frequent Itemsets: As you recursively mine the conditional FP-Trees, 
       you gather frequent itemsets. Each time a frequent itemset is found, 
       its added to the list of results.
    
    Applying The above concept to Lottery Number Gen: 
    1. Each lottery draw (i.e., the set of 6 numbers) is treated as a transaction in the dataset.
    2. FP-Growth analyzes these draws to find frequent combinations of numbers 
       that appear together more often than a specified threshold.
    3. Once frequent itemsets are found, they can be used to generate association rules, 
       which suggest how likely certain numbers are to appear together in future draws.
    '''
    
    # Step 1: Prepare data as transactions
    transactions = data[['wnum_1', 'wnum_2', 'wnum_3', 'wnum_4', 'wnum_5', 'wnum_6']].values.tolist()

    # Step 2: Transform the data
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_freq_itemsets = pd.DataFrame(te_ary, columns=te.columns_)

    # Step 3: Apply Apriori Algorithm
    # frequent_itemsets = apriori(df_apriori, min_support=0.12, use_colnames=True)
    min_support = 0.0001
    frequent_itemsets = fpgrowth(df_freq_itemsets, min_support=min_support, use_colnames=True)

    # Filter to get only pairs (length 2)
    frequent_pairs = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 2)]
    print(f"Frequent Pairs: \n{frequent_pairs}")

    frequent_triplets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 3)]
    print(f"Frequent Triplets: \n{frequent_triplets}")

    # Generate Association Rules with lower thresholds for confidence and lift
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
    significant_rules = rules[(rules['confidence'] > 0.5) & (rules['lift'] > 1) & (rules['support']>0.0008)]
    print("\nSignificant Rules:")
    print(significant_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction']])

    # Extract the antecedents (number combinations) from the rules
    antecedents = significant_rules['antecedents'].tolist()

    # Flatten the list of sets into a single set of numbers
    numbers_from_rules = set()
    for antecedent in antecedents:
        numbers_from_rules.update(antecedent)
    
    # Convert any NumPy integers to standard Python integers
    numbers_from_rules = {int(num) for num in numbers_from_rules}
    
    n_numbers = 6 # Number you want to generate for draw
    numbers_list = list(numbers_from_rules)
    print(numbers_list)
    # Randomly select numbers from the list
    if len(numbers_list) >= n_numbers:
        generated_draw = set(random.sample(numbers_list, n_numbers))
    else:
        generated_draw = set(numbers_list)
        # If fewer than n_numbers, add random numbers from the pool (1 to 49)
        all_numbers = set(range(1, 50))
        while len(generated_draw) < n_numbers:
            generated_draw.add(random.choice(list(all_numbers - generated_draw)))
        
    selected_numbers = generated_draw
    
    # Ensure the result is a list of integers
    return [int(num) for num in selected_numbers]


# estimate the number appearance then use weighted ranom selection to select the draw number. 
def weighted_random_pair_selection_generator(data, n_pairs=100, n_triplets=100):  
    ''' Weighted Random Pair Selection Design Steps: 
    - Extract all unique pairs of numbers from each draw (ignoring order).
    - Count the frequency of each pair across all historical draws.
    - Sort the pairs by their frequency of occurrence to identify the most common pairs.
    - Use the frequency data to generate future numbers. The pairs with higher frequencies will have a higher probability of being selected.
    - Ensure that the generator respects the rules of the lottery (e.g., no repetition of numbers within a draw).
    '''
    
    # Step 1: Extract all unique pairs and triplets from each draw
    pairs = []
    triplets = []
    for _, row in data.iterrows():
        numbers = [row[f'wnum_{i}'] for i in range(1, 7)]
        pairs.extend(combinations(sorted(numbers), 2))
        triplets.extend(combinations(sorted(numbers), 3))

    # Step 2: Count the frequency of each pair and triplet
    pair_counts = Counter(pairs)
    triplet_counts = Counter(triplets)

    # Print the top 10 pairs
    print("Top 10 Pairs:")
    for pair, count in pair_counts.most_common(10):
        print(f"Pair: {[int(n) for n in pair]}, Count: {count}")

    # Print the top 10 triplets
    print("\nTop 10 Triplets:")
    for triplet, count in triplet_counts.most_common(10):
        print(f"Triplet: {[int(n) for n in triplet]}, Count: {count}")

    # Flatten the Counters to create lists of items with weighted probabilities
    all_items = list(pair_counts.items()) + list(triplet_counts.items())
    items, weights = zip(*all_items)
    total = sum(weights)
    probabilities = [weight / total for weight in weights]

    # Randomly select items (pairs and triplets) based on their probabilities
    selected_items = random.choices(items, probabilities, k=n_pairs + n_triplets)

    # Flatten the selected items into a set of unique numbers
    selected_numbers = set()
    for item in selected_items:
        selected_numbers.update(item)
    print(f"NUMBER-POOLS:{[int(n) for n in selected_numbers]}")
    # Ensure exactly 6 numbers
    while len(selected_numbers) > 6:
        selected_numbers.remove(random.choice(list(selected_numbers)))
    
    # If fewer than 6 numbers, add random numbers not already in the set
    all_numbers = set(range(1, 50))
    while len(selected_numbers) < 6:
        selected_numbers.add(random.choice(list(all_numbers - selected_numbers)))
    
    # Return the selected numbers in Sorted order
    selected_numbers = sorted(selected_numbers)
    
    return [int(num) for num in selected_numbers]


# Combine the approach from FP Growth and Weighted Random
def adjusted_fp_growth_generator(data, logging=False):
    ''' Adjusted FP GROWTH 
    To improve the FP-growth method and estimate the latest draw using a pool created from 
    the top 6 frequent itemsets, pairs, and triplets
    Here are the updated design for number generator:
    -  Start by generating frequent itemsets, pairs, and triplets using FP-growth.
    -  From the frequent itemsets, pairs, and triplets, extract the top 6 items based on their frequency or support. 
       Combine these into a pool of potential numbers.
    -  Use a weighted or random selection method to pick 6 numbers from the pool created in prev step.
    
    '''

    # Step 1: Prepare data as transactions
    transactions = data[['wnum_1', 'wnum_2', 'wnum_3', 'wnum_4', 'wnum_5', 'wnum_6']].values.tolist()

    # Step 2: Transform the data
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_fpgrowth = pd.DataFrame(te_ary, columns=te.columns_)

    # Step 3: Apply FP-Growth Algorithm to generate frequent itemsets. 
    #         Use lowest min_support to include all itemsets.
    frequent_itemsets = fpgrowth(df_fpgrowth, min_support=0.0001, use_colnames=True)

    # Extract frequent pairs and triplets from the frequent itemsets
    frequent_pairs = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 2)]
    frequent_triplets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 3)]

    # Step 4: Create a number pool from the top 6 frequent itemsets, pairs, and triplets
    top_itemsets = frequent_itemsets.nlargest(6, 'support')['itemsets']
    top_pairs = frequent_pairs.nlargest(6, 'support')['itemsets']
    top_triplets = frequent_triplets.nlargest(6, 'support')['itemsets']

    # Combine all into a single pool
    number_pool = set()
    for itemset in pd.concat([top_itemsets, top_pairs, top_triplets]):
        number_pool.update(itemset)
    
    print(f"LATEST-NUMPOOL::{number_pool}") if logging else None

    # Step 5: Calculate Weights for Pairs and Triplets
    pair_weights = Counter()
    triplet_weights = Counter()

    for itemset in top_pairs:
        pair_weights.update(itemset)

    for itemset in top_triplets:
        triplet_weights.update(itemset)
    
    # Combine weights with a multiplier for pairs and triplets
    combined_weights = Counter()
    for num in number_pool:
        combined_weights[num] = (pair_weights[num] * 2) + (triplet_weights[num] * 3)
    
    print(f"COMBINED-WEIGHT::{combined_weights}") if logging else None
    
    # Step 6: Weighted Random Selection to Generate the Latest Draw
    n_numbers = 6
    pool = list(number_pool)
    total_weight = sum(combined_weights.values())
    probabilities = {num: weight / total_weight for num, weight in combined_weights.items()}
    
    selected_numbers = set()
    while len(selected_numbers) < n_numbers:
        selected = random.choices(list(pool), weights=[probabilities[num] for num in pool], k=1)[0]
        selected_numbers.add(selected)
        pool.remove(selected)  # Remove selected number from pool to avoid repetition
    
    final_selected_number = sorted(selected_numbers)
    
    return final_selected_number


