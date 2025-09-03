from CMCed.production_cycle import ProductionCycle
from CMCed.Cognitive_Functions import *
import pandas as pd
import matplotlib.pyplot as plt

#collect per-chunk utilities across all games/rounds
all_chunk_utilities = []  # holds per-round, per-chunk utilities for both agents

# Initialize individual memories for referee, agent1, and agent2

def run_single_game(num_iterations):


    referee_working_memory = {
        'focus_buffer': {'state': 'start'},
        'rounds_played': 0,
        'agent1_score': 0,
        'agent2_score': 0,
        'draws': 0,
        'score_difference': 0,
        'game_results': []
    }

    environment_memory = {
        'agent1hand': {'hand': 'open', 'move': 'unknown'},
        'agent2hand': {'hand': 'open', 'move': 'unknown'},
        'refinstructions': {'phase': 'start', 'agent1_done': 'not_done', 'agent2_done': 'not_done'}
    }

    agent1_working_memory = {
        'focus_buffer': {'state': 'start'},
        'lag_buffer': {'lag1': 'paper', 'lag0': 'unknown'}
    }

    agent1_declarative_memory = {
        'rp': {'lag1': 'rock', 'lag0': 'paper', 'utility': 1000},
        'rr': {'lag1': 'rock', 'lag0': 'rock', 'utility': 1000},
        'rs': {'lag1': 'rock', 'lag0': 'scissors', 'utility': 1000},
        'pp': {'lag1': 'paper', 'lag0': 'paper', 'utility': 1000},
        'pr': {'lag1': 'paper', 'lag0': 'rock', 'utility': 1000},
        'ps': {'lag1': 'paper', 'lag0': 'scissors', 'utility': 1000},
        'sp': {'lag1': 'scissors', 'lag0': 'paper', 'utility': 1000},
        'sr': {'lag1': 'scissors', 'lag0': 'rock', 'utility': 1000},
        'ss': {'lag1': 'scissors', 'lag0': 'scissors', 'utility': 1000}
    }

    agent2_working_memory = {
        'focus_buffer': {'state': 'start'},
        'lag_buffer': {'lag1': 'scissors', 'lag0': 'unknown'}
    }

    agent2_declarative_memory = {
        'rp': {'lag1': 'rock', 'lag0': 'paper', 'utility': 1000},
        'rr': {'lag1': 'rock', 'lag0': 'rock', 'utility': 1000},
        'rs': {'lag1': 'rock', 'lag0': 'scissors', 'utility': 1000},
        'pp': {'lag1': 'paper', 'lag0': 'paper', 'utility': 1000},
        'pr': {'lag1': 'paper', 'lag0': 'rock', 'utility': 1000},
        'ps': {'lag1': 'paper', 'lag0': 'scissors', 'utility': 1000},
        'sp': {'lag1': 'scissors', 'lag0': 'paper', 'utility': 1000},
        'sr': {'lag1': 'scissors', 'lag0': 'rock', 'utility': 1000},
        'ss': {'lag1': 'scissors', 'lag0': 'scissors', 'utility': 1000}
    }

    # memory dictionary
    memories = {
        'agent1_working_memory': agent1_working_memory,
        'agent2_working_memory': agent2_working_memory,
        'environment_memory': environment_memory,
        'agent1_declarative_memory': agent1_declarative_memory,
        'agent2_declarative_memory': agent2_declarative_memory,
        'referee_working_memory': referee_working_memory
    }

    # Procedural Productions
    Agent1Productions = []
    Agent2Productions = []
    ProceduralProductions = []





    #################################################################################################################

    # Start game production (Referee initiates the game)
    def start_game(memories):
        ref_mem = memories['referee_working_memory']
        ref_mem['rounds_played'] += 1
        print(f"\nStarting Round {ref_mem['rounds_played']}")
        memories['environment_memory']['refinstructions']['phase'] = 'play_round'
        ref_mem['focus_buffer']['state'] = 'waiting_for_round'
        print(f"Referee Focus: {memories['referee_working_memory']['focus_buffer']['state']}") 
        print(f"Environment Phase: {memories['environment_memory']['refinstructions']['phase']}")

    ProceduralProductions.append({
        'matches': {'referee_working_memory': {'focus_buffer': {'state': 'start'}}},
        'negations': {},
        'utility': 10,
        'action': start_game,
        'report': "start_game"
    })

    # Referee ensures both agents have checked moves before proceeding to evaluation
    def check_agents_done_round(memories):
        print("Referee confirms both agents have finished the round.")
        memories['referee_working_memory']['focus_buffer']['state'] = 'evaluate_results'
        memories['environment_memory']['refinstructions']['agent1_done'] = 'not_done'
        memories['environment_memory']['refinstructions']['agent2_done'] = 'not_done'

    ProceduralProductions.append({
        'matches': {'environment_memory': {'refinstructions': {'agent1_done': 'done2', 'agent2_done': 'done2'}}},
        'negations': {},
        'utility': 10,
        'action': check_agents_done_round,
        'report': "check_agents_done_round"
    })

    # Evaluation and scoring
    def evaluate_results(memories):
        ref_mem = memories['referee_working_memory']
        agent1_choice = memories['environment_memory']['agent1hand']['move']
        agent2_choice = memories['environment_memory']['agent2hand']['move']

        # Calculate average utility for each agent's declarative memory
        agent1_utility = (
            sum(chunk['utility'] for chunk in memories['agent1_declarative_memory'].values()) /
            len(memories['agent1_declarative_memory'])
        )
        agent2_utility = (
            sum(chunk['utility'] for chunk in memories['agent2_declarative_memory'].values()) /
            len(memories['agent2_declarative_memory'])
        )

        
        if agent1_choice == agent2_choice:
            result = "It's a tie!"
            ref_mem['draws'] += 1
        elif (agent1_choice == 'rock' and agent2_choice == 'scissors') or \
            (agent1_choice == 'paper' and agent2_choice == 'rock') or \
            (agent1_choice == 'scissors' and agent2_choice == 'paper'):
            result = "Agent 1 wins!"
            ref_mem['agent1_score'] += 1
        else:
            result = "Agent 2 wins!"
            ref_mem['agent2_score'] += 1

        ref_mem['score_difference'] = ref_mem['agent1_score'] - ref_mem['agent2_score']
        print(f"Referee sees that Agent 1 played {agent1_choice} and Agent 2 played {agent2_choice}. {result}")
        print(f"Current Score - Agent 1: {ref_mem['agent1_score']}, Agent 2: {ref_mem['agent2_score']}, Draws: {ref_mem['draws']}, Score Difference: {ref_mem['score_difference']}")

        #snapshot per-chunk utilities for this round (both agents)
        curr_round = ref_mem['rounds_played']
        for chunk_key, chunk_val in memories['agent1_declarative_memory'].items():
            all_chunk_utilities.append({
                'Game': num_iterations,
                'Round': curr_round,
                'Agent': 'Agent 1',
                'Chunk': chunk_key,
                'Utility': chunk_val['utility']
            })
        for chunk_key, chunk_val in memories['agent2_declarative_memory'].items():
            all_chunk_utilities.append({
                'Game': num_iterations,
                'Round': curr_round,
                'Agent': 'Agent 2',
                'Chunk': chunk_key,
                'Utility': chunk_val['utility']
            })
        
        round_data = {
            'Round': ref_mem['rounds_played'],
            'Agent 1 Score': ref_mem['agent1_score'],
            'Agent 2 Score': ref_mem['agent2_score'],
            'Draws': ref_mem['draws'],
            'Score Difference': ref_mem['score_difference'],
            'Agent 1 Utility': agent1_utility,
            'Agent 2 Utility': agent2_utility,
            'Result': result
        }
        ref_mem['game_results'].append(round_data)
        ref_mem['focus_buffer']['state'] = 'end_game'

    ProceduralProductions.append({
        'matches': {'referee_working_memory': {'focus_buffer': {'state': 'evaluate_results'}}},
        'negations': {},
        'utility': 10,
        'action': evaluate_results,
        'report': "evaluate_results"
    })

    # End game and prepare for next round
    def end_game(memories):
        ref_mem = memories['referee_working_memory']
        print(f"Ending Round {ref_mem['rounds_played']}")
        ref_mem['focus_buffer']['state'] = 'start'
        agent1_working_memory['focus_buffer']['state'] = 'start'
        agent2_working_memory['focus_buffer']['state'] = 'start'


    ProceduralProductions.append({
        'matches': {'referee_working_memory': {'focus_buffer': {'state': 'end_game'}}},
        'negations': {},
        'utility': 10,
        'action': end_game,
        'report': "end_game"
    })



    #######################################################################################################
    # Agent1Productions

    def recall_agent1_move(memories):
        print('111111111111111111111111111111111111')
        # decay agent1_declarative_memory
        decay_all_memory_chunks(memories, 'agent1_declarative_memory', 1)
        #add_noise_to_utility(agent1_declarative_memory, 'agent1_declarative_memory', scalar=1)
        # get lag1 from lag_buffer
        lag1 = memories['agent1_working_memory']['lag_buffer']['lag1']
        print('lag_buffer is: ', memories['agent1_working_memory']['lag_buffer'])
        # use lag1 to retrieve prediction from agent1_declarative_memory
        target_memory = memories['agent1_declarative_memory']
        matches = {'lag1': lag1, 'lag0': '*'}  # already shifted lag1
        negations = {}
        retrieved_chunk = retrieve_memory_chunk(target_memory, matches, negations)
        print(f"Agent 1 recalls {retrieved_chunk}")

        memories['agent1_working_memory']['lag_buffer'] = retrieved_chunk
        print('lag_buffer updated to :', memories['agent1_working_memory']['lag_buffer'])
        memories['agent1_working_memory']['focus_buffer']['state'] = 'counter'



    Agent1Productions.append({
        'matches': {'agent1_working_memory': {'focus_buffer': {'state': 'start'}}},
        'negations': {},
        'utility': 10,
        'action': recall_agent1_move,
        'report': "agent1_recall_move"
    })


    ### get counter move

    # Agent 1 counters 'paper' by choosing 'scissors'
    def counter_paper_agent1(memories):
        print('111111111111111111111111111111111111')

        memories['environment_memory']['agent1hand']['move'] = 'scissors'
        print(environment_memory)

        print("Agent 1 chooses 'scissors' to counter paper")
        memories['agent1_working_memory']['focus_buffer']['state'] = 'check_move'



    Agent1Productions.append({
        'matches': {'agent1_working_memory': {'focus_buffer': {'state': 'counter'}, 'lag_buffer': {'lag0': 'paper'}}},
        'negations': {},
        'utility': 10,
        'action': counter_paper_agent1,
        'report': "counter_paper_agent1"
    })

    # Agent 1 counters 'scissors' by choosing 'rock'
    def counter_scissors_agent1(memories):
        print('111111111111111111111111111111111111')
        memories['environment_memory']['agent1hand']['move'] = 'rock'
        print(environment_memory)
        print("Agent 1 chooses 'rock' to counter scissors")
        memories['agent1_working_memory']['focus_buffer']['state'] = 'check_move'



    Agent1Productions.append({
        'matches': {'agent1_working_memory': {'focus_buffer': {'state': 'counter'}, 'lag_buffer': {'lag0': 'scissors'}}},
        'negations': {},
        'utility': 10,
        'action': counter_scissors_agent1,
        'report': "counter_scissors_agent1"
    })

    # Agent 1 counters 'rock' by choosing 'paper'
    def counter_rock_agent1(memories):
        print('111111111111111111111111111111111111')
        memories['environment_memory']['agent1hand']['move'] = 'paper'
        print(environment_memory)

        print("Agent 1 chooses 'paper' to counter rock")
        memories['agent1_working_memory']['focus_buffer']['state'] = 'check_move'


    Agent1Productions.append({
        'matches': {'agent1_working_memory': {'focus_buffer': {'state': 'counter'}, 'lag_buffer': {'lag0': 'rock'}}},
        'negations': {},
        'utility': 10,
        'action': counter_rock_agent1,
        'report': "counter_rock_agent1"
    })

    # Agent 1 checks Agent 2's move

    def agent1_checks_agent2_move(memories):
        print('111111111111111111111111111111111111')
        opponent_move = memories['environment_memory']['agent2hand']['move']
        print(f"Agent 1 sees that Agent 2 played: {opponent_move}")

        # update lag buffer to true state
        memories['agent1_working_memory']['lag_buffer']['lag0'] = opponent_move
        print(f"updated lag buffer for Agent 1 to actual result: {memories['agent1_working_memory']['lag_buffer']}")
        # boost the chunk in memory that corresponds to the chunk in the lag_buffer
        chunk_description = memories['agent1_working_memory']['lag_buffer']
        # Print the chunk description for clarity
        print("Extracted chunk description:", chunk_description)
        # Boost the utility
        utility_change_by_description(memories, 'agent1_declarative_memory', chunk_description, amount=1, max_utility=10000)
        # shift and update lag buffer for next guess
        memories['agent1_working_memory']['lag_buffer']['lag1'] = opponent_move
        memories['agent1_working_memory']['lag_buffer']['lag0'] = 'unknown'
        print(f"shifted lag buffer for Agent 1 for next round: {memories['agent1_working_memory']['lag_buffer']}")
        memories['environment_memory']['refinstructions']['agent1_done'] = 'done2'
        memories['agent1_working_memory']['focus_buffer']['state'] = 'waiting2'


    Agent1Productions.append({
        'matches': {'agent1_working_memory': {'focus_buffer': {'state': 'check_move'}},
                    'environment_memory': {'agent2hand': {'hand': 'open', 'move': '*'}},
                    'environment_memory': {'refinstructions': {'phase': 'play_round'}}},
        'negations': {},
        'utility': 10,
        'action': agent1_checks_agent2_move,
        'report': "check_agent2_move"
    })


    #####################################################################################################
    # Agent2Productions

    def recall_agent2_move(memories):
        print('222222222222222222222222222222222222')
        # decay agent1_declarative_memory
        decay_all_memory_chunks(memories, 'agent2_declarative_memory', 1)
        #add_noise_to_utility(agent2_declarative_memory, 'agent2_declarative_memory', scalar=1)
        # get lag1 from lag_buffer
        lag1 = memories['agent2_working_memory']['lag_buffer']['lag1']
        print('lag_buffer is: ', memories['agent2_working_memory']['lag_buffer'])
        # use lag1 to retrieve prediction from agent1_declarative_memory
        target_memory = memories['agent2_declarative_memory']
        matches = {'lag1': lag1, 'lag0': '*'}  # already shifted lag1
        negations = {}
        retrieved_chunk = retrieve_memory_chunk(target_memory, matches, negations)
        print(f"Agent 2 recalls {retrieved_chunk}")
        memories['agent2_working_memory']['lag_buffer'] = retrieved_chunk
        print('lag_buffer updated to :', memories['agent2_working_memory']['lag_buffer'])
        memories['agent2_working_memory']['focus_buffer']['state'] = 'counter'

    Agent2Productions.append({
        'matches': {'agent2_working_memory': {'focus_buffer': {'state': 'start'}}},
        'negations': {},
        'utility': 10,
        'action': recall_agent2_move,
        'report': "agent2_recall_move"
    })


    ### get counter move
    def counter_paper_agent2(memories):
        print('222222222222222222222222222222222222')
        memories['environment_memory']['agent2hand']['move'] = 'scissors'
        print(environment_memory)

        print("Agent 2 chooses 'scissors' to counter paper")
        memories['agent2_working_memory']['focus_buffer']['state'] = 'check_move'


    Agent2Productions.append({
        'matches': {'agent2_working_memory': {'focus_buffer': {'state': 'counter'}, 'lag_buffer': {'lag0': 'paper'}}},
        'negations': {},
        'utility': 10,
        'action': counter_paper_agent2,
        'report': "counter_paper_agent2"
    })

    def counter_scissors_agent2(memories):
        print('222222222222222222222222222222222222')
        memories['environment_memory']['agent2hand']['move'] = 'rock'
        print(environment_memory)

        print("Agent 2 chooses 'rock' to counter scissors")
        memories['agent2_working_memory']['focus_buffer']['state'] = 'check_move'

    Agent2Productions.append({
        'matches': {'agent2_working_memory': {'focus_buffer': {'state': 'counter'}, 'lag_buffer': {'lag0': 'scissors'}}},
        'negations': {},
        'utility': 10,
        'action': counter_scissors_agent2,
        'report': "counter_scissors_agent2"
    })

    def counter_rock_agent2(memories):
        print('222222222222222222222222222222222222')
        memories['environment_memory']['agent2hand']['move'] = 'paper'
        print(environment_memory)

        print("Agent 2 chooses 'paper' to counter rock")
        memories['agent2_working_memory']['focus_buffer']['state'] = 'check_move'

    Agent2Productions.append({
        'matches': {'agent2_working_memory': {'focus_buffer': {'state': 'counter'}, 'lag_buffer': {'lag0': 'rock'}}},
        'negations': {},
        'utility': 10,
        'action': counter_rock_agent2,
        'report': "counter_rock_agent2"
    })


    # Agent 2 checks Agent 1's move

    def agent2_checks_agent1_move(memories):
        print('222222222222222222222222222222222222')
        opponent_move = memories['environment_memory']['agent1hand']['move']
        print(f"Agent 2 sees that Agent 1 played: {opponent_move}")

        # update lag buffer to true state
        memories['agent2_working_memory']['lag_buffer']['lag0'] = opponent_move
        print(f"updated lag buffer for Agent 2 to actual result: {memories['agent2_working_memory']['lag_buffer']}")
        # boost the chunk in memory that corresponds to the chunk in the lag_buffer
        chunk_description = memories['agent2_working_memory']['lag_buffer']
        # Print the chunk description for clarity
        print("Extracted chunk description:", chunk_description)
        # Boost the utility
        utility_change_by_description(memories, 'agent2_declarative_memory', chunk_description, amount=1, max_utility=10000)
        # shift and update lag buffer for next guess
        memories['agent2_working_memory']['lag_buffer']['lag1'] = opponent_move
        memories['agent2_working_memory']['lag_buffer']['lag0'] = 'unknown'
        print(f"shifted lag buffer for Agent 2 for next round: {memories['agent2_working_memory']['lag_buffer']}")
        memories['environment_memory']['refinstructions']['agent2_done'] = 'done2'
        memories['agent2_working_memory']['focus_buffer']['state'] = 'waiting2'


    Agent2Productions.append({
        'matches': {'agent2_working_memory': {'focus_buffer': {'state': 'check_move'}},
                    'environment_memory': {'agent1hand': {'hand': 'open', 'move': '*'}},
                    'environment_memory': {'refinstructions': {'phase': 'play_round'}}},
        'negations': {},
        'utility': 10,
        'action': agent2_checks_agent1_move,
        'report': "check_agent2_move"
    })



    # Production system delays in ticks
    ProductionSystem_Countdown = 1
    Agent1Productions_Countdown = 1
    Agent2Productions_Countdown = 1


    # Stores the number of cycles for a production system to fire and reset

    DelayResetValues = {
        'ProductionSystem1': ProductionSystem_Countdown,
        'Agent1Productions': Agent1Productions_Countdown,
        'Agent2Productions': Agent2Productions_Countdown
    }

    # Dictionary of all production systems and delays

    AllProductionSystems = {
        'ProductionSystem1': [ProceduralProductions, ProductionSystem_Countdown],
        'Agent1Productions': [Agent1Productions, Agent1Productions_Countdown],
        'Agent2Productions': [Agent2Productions, Agent2Productions_Countdown]
    }

    # Initialize ProductionCycle

    ps = ProductionCycle()

    # Run the cycle with custom parameters

    ps.run_cycles(memories, AllProductionSystems, DelayResetValues, cycles=12000, millisecpercycle=10)


        # Collect results into all_results
    game_results = pd.DataFrame(referee_working_memory['game_results'])
    game_results['Game'] = num_iterations  # Add a column to identify the game
    all_results.append(game_results)


all_results = []

    # Loop to run multiple games
num_iterations = 2 # Number of games
for i in range(1, num_iterations + 1):
    print(f"\n=== Starting Game {i} ===")
    run_single_game(i)
    print(f"=== Ending Game {i} ===")



# Combine all results into a single DataFrame
combined_results = pd.concat(all_results, ignore_index=True)

# Calculate overall averages across all games
overall_avg_score_difference = combined_results['Score Difference'].mean()
overall_avg_utility_agent1 = combined_results['Agent 1 Utility'].mean()
overall_avg_utility_agent2 = combined_results['Agent 2 Utility'].mean()

# Print the overall average metrics
print(" Overall Average Metrics Across All Games:")
print(f"  - Average Score Difference: {overall_avg_score_difference:.2f}")
print(f"  - Average Utility for Agent 1: {overall_avg_utility_agent1:.2f}")
print(f"  - Average Utility for Agent 2: {overall_avg_utility_agent2:.2f}")

# Create a summary DataFrame for overall averages
summary_data = pd.DataFrame({
    "Metric": ["Average Score Difference", "Average Utility (Agent 1)", "Average Utility (Agent 2)"],
    "Value": [overall_avg_score_difference, overall_avg_utility_agent1, overall_avg_utility_agent2]
})

# Calculate per-game averages
game_averages = combined_results.groupby('Game')[['Score Difference', 'Agent 1 Utility', 'Agent 2 Utility']].mean().reset_index()
game_averages.rename(columns={
    "Score Difference": "Avg Score Difference",
    "Agent 1 Utility": "Avg Agent 1 Utility",
    "Agent 2 Utility": "Avg Agent 2 Utility"
}, inplace=True)

# Print per-game average metrics
print("Per-Game Average Metrics:")
for _, row in game_averages.iterrows():
    print(f"  Game {int(row['Game'])}:")
    print(f"      - Avg Score Difference: {row['Avg Score Difference']:.2f}")
    print(f"      - Avg Agent 1 Utility: {row['Avg Agent 1 Utility']:.2f}")
    print(f"      - Avg Agent 2 Utility: {row['Avg Agent 2 Utility']:.2f}")

# adds averaged chunk utilities
chunk_util_df = pd.DataFrame(all_chunk_utilities)

# Average utility per Agent × Chunk × Round across all games
avg_chunk_util = (
    chunk_util_df
    .groupby(['Agent', 'Chunk', 'Round'], as_index=False)['Utility']
    .mean()
)

# Save both the detailed results, per-game averages, and overall summary in one Excel file
excel_filename = "Results.xlsx"
with pd.ExcelWriter(excel_filename, engine="xlsxwriter") as writer:
    combined_results.to_excel(writer, sheet_name="Game Results", index=False)  # Saves the full game results (all columns)
    game_averages.to_excel(writer, sheet_name="Game Averages", index=False)  # Saves per-game averages
    summary_data.to_excel(writer, sheet_name="Summary", index=False)  # Appends overall summary

print(f"All game results and average metrics saved to '{excel_filename}'.")


# Plot a graph for each game
for game_number in combined_results['Game'].unique():
    game_data = combined_results[combined_results['Game'] == game_number]

        # Calculate average utility for the game
    avg_utility_agent1 = game_data['Agent 1 Utility'].mean()
    avg_utility_agent2 = game_data['Agent 2 Utility'].mean()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot the score difference (primary y-axis)
    ax1.plot(game_data['Round'], game_data['Score Difference'], linestyle='-', color='b', label='Score Difference')
    ax1.fill_between(game_data['Round'], game_data['Score Difference'], 
                     where=(game_data['Score Difference'] > 0), color='green', alpha=0.3, label="Agent 1 Leading")
    ax1.fill_between(game_data['Round'], game_data['Score Difference'], 
                     where=(game_data['Score Difference'] < 0), color='red', alpha=0.3, label="Agent 2 Leading")
    ax1.axhline(0, color='black', linewidth=1, linestyle='--')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Score Difference')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot the average utilities for each agent (secondary y-axis)
    ax2 = ax1.twinx()
    ax2.plot(game_data['Round'], game_data['Agent 1 Utility'], linestyle='--', color='teal', label='Agent 1 Utility')
    ax2.plot(game_data['Round'], game_data['Agent 2 Utility'], linestyle='--', color='#E44D26', label='Agent 2 Utility')
    ax2.set_ylabel('Utility')
    ax2.legend(loc='upper right')

    # Title and save
    plt.title(f'Score Difference and Utilities (Game {game_number})')
    plt.savefig(f'Score Difference and Utilities{game_number}.png')  
    plt.show()

# Calculate the average score difference across all games
average_score_difference = combined_results.groupby('Round')['Score Difference'].mean()

# Calculate average utilities for each agent across all games
average_utilities = combined_results.groupby('Round')[['Agent 1 Utility', 'Agent 2 Utility']].mean()

# Plot average score difference and utilities on the same plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the average score difference (primary y-axis)
ax1.plot(average_score_difference.index, average_score_difference.values, 
         linestyle='-', color='blue', label='Average Score Difference')
ax1.fill_between(average_score_difference.index, average_score_difference.values, 
                 where=(average_score_difference.values > 0), color='green', alpha=0.3, label="Agent 1 Leading")
ax1.fill_between(average_score_difference.index, average_score_difference.values, 
                 where=(average_score_difference.values < 0), color='red', alpha=0.3, label="Agent 2 Leading")
ax1.axhline(0, color='black', linewidth=1, linestyle='--')
ax1.set_xlabel('Round')
ax1.set_ylabel('Score Difference')
ax1.legend(loc='upper left')
ax1.grid(True)

# Plot the average utilities for each agent (secondary y-axis)
ax2 = ax1.twinx()
ax2.plot(average_utilities.index, average_utilities['Agent 1 Utility'], linestyle='--', color='teal', label='Avg Agent 1 Utility')
ax2.plot(average_utilities.index, average_utilities['Agent 2 Utility'], linestyle='--', color='#E44D26', label='Avg Agent 2 Utility')
ax2.set_ylabel('Utility')
ax2.legend(loc='upper right')

# Title and save
plt.title('Average Score Difference and Utilities Across Games')
plt.savefig('Average Score Difference and Utilities Across Games.png')  
plt.show()

# Loop through each game and plot its score difference
for game_number in combined_results['Game'].unique():
    game_data = combined_results[combined_results['Game'] == game_number]
    plt.plot(game_data['Round'], game_data['Score Difference'], label=f'Game {game_number}')

# Labels, title, and legend
plt.xlabel('Round')
plt.ylabel('Score Difference')
plt.title('Score Difference Across All Games')
plt.axhline(0, color='black', linewidth=1, linestyle='--')  
plt.legend(title="Games", loc='upper right')  
plt.grid(True)

# Save and show the plot
plt.savefig('score_difference_all_games.png')  
plt.show()

# Average per-chunk utility across ALL games (de-overlap equal values per round) ===
# Requires: chunk_util_df built from all_chunk_utilities

if 'chunk_util_df' in globals() and not chunk_util_df.empty:
    # 1) Average by Agent × Chunk × Round across games
    avg_chunk_util = (
        chunk_util_df
        .groupby(['Agent', 'Chunk', 'Round'], as_index=False)['Utility']
        .mean()
    )

    def plot_avg_with_tie_jitter(agent_label, jitter_amplitude=0.4):
        """
        Plot all chunks on one figure with small vertical jitter applied
        ONLY to ties (chunks that have the exact same value in a round).
        jitter_amplitude is in 'utility' units (e.g., 0.4).
        """
        sub = avg_chunk_util[avg_chunk_util['Agent'] == agent_label]
        if sub.empty:
            return

        pivot = sub.pivot(index='Round', columns='Chunk', values='Utility').sort_index()
        rounds = list(pivot.index)
        chunks = list(pivot.columns)

        # Build adjusted series: for each round, separate chunks with identical values
        adjusted = {chunk: [] for chunk in chunks}

        for r in rounds:
            # group chunks by their (possibly identical) value at this round
            value_to_chunks = {}
            for chunk in chunks:
                val = pivot.at[r, chunk]
                value_to_chunks.setdefault(val, []).append(chunk)

            # assign small symmetric offsets within each tie group
            for val, group in value_to_chunks.items():
                m = len(group)
                if m == 1:
                    offsets = [0.0]
                else:
                    # spread offsets in [-0.5, +0.5], symmetric
                    if m == 2:
                        offsets = [-0.5, 0.5]
                    else:
                        step = 1.0 / (m - 1)  # 0..1
                        offsets = [i * step - 0.5 for i in range(m)]
                for chunk, off in zip(group, offsets):
                    adjusted[chunk].append(val + jitter_amplitude * off)

        # Plot
        plt.figure(figsize=(12, 7))
        for chunk in chunks:
            plt.plot(rounds, adjusted[chunk], linewidth=1.4, alpha=0.95, label=chunk, marker='.', markersize=2)

        plt.title(f'{agent_label} Chunk Utilities Over Rounds (Average Across Games)')
        plt.xlabel('Round')
        plt.ylabel('Average Utility')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Chunk', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        outname = f'chunk_utilities_avg_{agent_label.replace(" ", "").lower()}_deoverlap.png'
        plt.savefig(outname, dpi=300)
        plt.show()

    # Make the two figures (one per agent)
    plot_avg_with_tie_jitter('Agent 1', jitter_amplitude=0.4)
    plot_avg_with_tie_jitter('Agent 2', jitter_amplitude=0.4)