use crate::nn_one_hidden_layer;
use rand::Rng;
use std::env;
use std::io::{self, BufRead};

/// Neural network parameters.
const ROWS_SIZE: usize = 4;
const BOARD_SIZE: usize = ROWS_SIZE * ROWS_SIZE;
const NN_INPUT_SIZE: usize = BOARD_SIZE * 2;
const NN_HIDDEN_SIZE: usize = 210;
const NN_OUTPUT_SIZE: usize = ROWS_SIZE * ROWS_SIZE;
const LEARNING_RATE: f64 = 0.05; // Reduced from 0.1 for more stable learning

/// Game board structure.
struct GameState {
    board: [char; BOARD_SIZE],
    current_player: bool, // False for human (X), True for computer (O)
}

/// Neural network structure. For simplicity we have just
/// one hidden layer and fixed sizes (see defines above).
/// However for this problem going deeper than one hidden layer
/// is useless.
pub struct NeuralNetwork {
    weights_ih: [f64; NN_INPUT_SIZE * NN_HIDDEN_SIZE],
    weights_hh: [f64; NN_HIDDEN_SIZE * NN_HIDDEN_SIZE],
    weights_ho: [f64; NN_HIDDEN_SIZE * NN_OUTPUT_SIZE],
    biases_h: [f64; NN_HIDDEN_SIZE],
    biases_h2: [f64; NN_HIDDEN_SIZE],
    biases_o: [f64; NN_OUTPUT_SIZE],

    inputs: [f64; NN_INPUT_SIZE],
    hidden: [f64; NN_HIDDEN_SIZE],
    hidden2: [f64; NN_HIDDEN_SIZE],
    raw_logits: [f64; NN_OUTPUT_SIZE], // Output before softmax()
    outputs: [f64; NN_OUTPUT_SIZE],    // Output after softmax()
}

/// ReLU activation function.
fn relu(x: f64) -> f64 {
    if x > 0.0 {
        return x;
    }

    return 0.0;
}

/// ReLU derivative function.
fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        return 1.0;
    }

    return 0.0;
}

/// random_weight returns a random float number to be used to initialize the NN weights.
fn random_weight() -> f64 {
    let mut rng = rand::rng();
    rng.r#random::<f64>() - 0.5
}

/// Initialize the NN
pub fn init_nn() -> NeuralNetwork {
    let mut nn = NeuralNetwork {
        weights_ih: [0.0; NN_INPUT_SIZE * NN_HIDDEN_SIZE],
        weights_hh: [0.0; NN_HIDDEN_SIZE * NN_HIDDEN_SIZE],
        weights_ho: [0.0; NN_HIDDEN_SIZE * NN_OUTPUT_SIZE],
        biases_h: [0.0; NN_HIDDEN_SIZE],
        biases_h2: [0.0; NN_HIDDEN_SIZE],
        biases_o: [0.0; NN_OUTPUT_SIZE],

        inputs: [0.0; NN_INPUT_SIZE],
        hidden: [0.0; NN_HIDDEN_SIZE],
        hidden2: [0.0; NN_HIDDEN_SIZE],
        raw_logits: [0.0; NN_OUTPUT_SIZE],
        outputs: [0.0; NN_OUTPUT_SIZE],
    };

    for i in 0..(NN_INPUT_SIZE * NN_HIDDEN_SIZE) {
        nn.weights_ih[i] = random_weight();
    }

    for i in 0..(NN_HIDDEN_SIZE * NN_HIDDEN_SIZE) {
        nn.weights_hh[i] = random_weight();
    }

    for i in 0..(NN_HIDDEN_SIZE * NN_OUTPUT_SIZE) {
        nn.weights_ho[i] = random_weight();
    }

    for i in 0..NN_HIDDEN_SIZE {
        nn.biases_h[i] = random_weight();
    }

    for i in 0..NN_HIDDEN_SIZE {
        nn.biases_h2[i] = random_weight();
    }

    for i in 0..NN_OUTPUT_SIZE {
        nn.biases_o[i] = random_weight()
    }

    return nn;
}

/// Convert board state to neural network inputs. Note that we use
/// a peculiar encoding antirez descrived here:
/// https://www.youtube.com/watch?v=EXbgUXt8fFU
///
/// Instead of one-hot encoding, we can represent N different categories
/// as different bit patterns. In this specific case it's trivial:
///
/// 00 = empty
/// 10 = X
/// 01 = O
///
/// Two inputs per symbol instead of 3 in this case, but in the general case
/// this reduces the input dimensionality A LOT.
///
/// LEARNING OPPORTUNITY: You may want to learn (if not already aware) of
/// different ways to represent non scalar inputs in neural networks:
/// One hot encoding, learned embeddings, and even if it's just my random
/// exeriment this "permutation coding" that I'm using here.
///
fn board_to_inputs(state: &GameState, inputs: &mut [f64; NN_INPUT_SIZE]) {
    for i in 0..BOARD_SIZE {
        if state.board[i] == '.' {
            inputs[i * 2] = 0.0;
            inputs[(i * 2) + 1] = 0.0;
        } else if state.board[i] == 'X' {
            inputs[i * 2] = 1.0;
            inputs[(i * 2) + 1] = 0.0;
        } else {
            inputs[i * 2] = 0.0;
            inputs[(i * 2) + 1] = 1.0;
        }
    }
}

/// Neural network foward pass (inference). We store the activations
/// so we can also do backpropagation later.
fn forward_pass(nn: &mut NeuralNetwork, inputs: [f64; NN_INPUT_SIZE]) {
    nn.inputs = inputs;

    // Input to hidden layer.
    for i in 0..NN_HIDDEN_SIZE {
        let mut sum = nn.biases_h[i];
        for j in 0..NN_INPUT_SIZE {
            sum += inputs[j] * nn.weights_ih[j * NN_HIDDEN_SIZE + i];
        }
        nn.hidden[i] = relu(sum);
    }

    // hidden layer to hidden 2 layer.
    for i in 0..NN_HIDDEN_SIZE {
        let mut sum = nn.biases_h2[i];
        for j in 0..NN_HIDDEN_SIZE {
            sum += nn.hidden[j] * nn.weights_hh[j * NN_HIDDEN_SIZE + i];
        }
        nn.hidden2[i] = relu(sum);
    }

    // Hidden to output (raw logits).
    for i in 0..NN_OUTPUT_SIZE {
        nn.raw_logits[i] = nn.biases_o[i];
        for j in 0..NN_HIDDEN_SIZE {
            nn.raw_logits[i] += nn.hidden2[j] * nn.weights_ho[j * NN_OUTPUT_SIZE + i];
        }
    }

    // Apply softmax to get the final probabilities.
    softmax(nn);
}

fn softmax(nn: &mut NeuralNetwork) {
    let inputs = nn.raw_logits; // logits

    // Note: Softmax uses e(zj) which can overflow for big zj. To avoid this issue, we first find the max(zj) in inputs,
    // then we subtract max(zj) from all in z

    // Find the max input
    let mut m = inputs[0];
    for i in 1..NN_OUTPUT_SIZE {
        if inputs[i] > m {
            m = inputs[i];
        }
    }

    // Subtract m from the inputs.
    let mut sum_exp = 0.0; // to use for softmax
    for i in 0..NN_OUTPUT_SIZE {
        let adjusted = inputs[i] - m;
        sum_exp += libm::exp(adjusted);
    }

    if sum_exp > 0.0 {
        // Calculate SoftMax
        for i in 0..NN_OUTPUT_SIZE {
            nn.outputs[i] = libm::exp(inputs[i] - m) / sum_exp;
        }
    } else {
        // Fallback in case of numerical issues, just provide
        // a uniform distribution.
        for i in 0..NN_OUTPUT_SIZE {
            nn.outputs[i] = 1.0 / (NN_OUTPUT_SIZE as f64);
        }
    }
}

/// Get the best move computer move using the neural network.
/// Note that there is no complex sampling at all, we just get
/// the output with the highest value that has an empty tile.
fn play_computer_move(state: &GameState, nn: &mut NeuralNetwork, display_move: bool) -> usize {
    let mut inputs = [0.0; NN_INPUT_SIZE];

    board_to_inputs(state, &mut inputs);
    forward_pass(nn, inputs);

    // Find the highest probability value and best legal move.
    let mut highest_prob: f64 = -1.0;
    let mut highest_prob_idx: i8 = -1;
    let mut best_move: i8 = -1;
    let mut best_legal_prob: f64 = -1.0;

    for i in 0..NN_OUTPUT_SIZE {
        // Track highest probability overall.
        if nn.outputs[i] > highest_prob {
            highest_prob = nn.outputs[i];
            highest_prob_idx = i as i8;
        }

        // Track best legal move.
        if state.board[i] == '.' && (best_move == -1 || nn.outputs[i] > best_legal_prob) {
            best_move = i as i8;
            best_legal_prob = nn.outputs[i];
        }
    }

    if display_move {
        print!("Neural network move probabilities:\n");
        let mut pos;
        for row in 0..ROWS_SIZE {
            for col in 0..ROWS_SIZE {
                pos = row * ROWS_SIZE + col;

                // Print probability as percentage.
                print!("{:.1}%", nn.outputs[pos] * 100.0);

                // Add markers.
                if (pos as i8) == highest_prob_idx {
                    print!("*"); // Highest probability overall.
                }
                if (pos as i8) == best_move {
                    print!("#"); // Selected move (highest valid probability).
                }
                print!(" ");
            }
            println!("");
        }
    }

    return best_move as usize;
}


// To use only during match against other NN
fn board_to_inputs_nn(state: &nn_one_hidden_layer::GameState, inputs: &mut [f64; NN_INPUT_SIZE]) {
    for i in 0..BOARD_SIZE {
        if state.board[i] == '.' {
            inputs[i * 2] = 0.0;
            inputs[(i * 2) + 1] = 0.0;
        } else if state.board[i] == 'X' {
            inputs[i * 2] = 1.0;
            inputs[(i * 2) + 1] = 0.0;
        } else {
            inputs[i * 2] = 0.0;
            inputs[(i * 2) + 1] = 1.0;
        }
    }
}

// To use only during match against other NN
pub fn play_computer_move_nn(state: &nn_one_hidden_layer::GameState, nn: &mut NeuralNetwork, display_move: bool) -> usize {
    let mut inputs = [0.0; NN_INPUT_SIZE];

    board_to_inputs_nn(state, &mut inputs);
    forward_pass(nn, inputs);

    // Find the highest probability value and best legal move.
    let mut highest_prob: f64 = -1.0;
    let mut highest_prob_idx: i8 = -1;
    let mut best_move: i8 = -1;
    let mut best_legal_prob: f64 = -1.0;

    for i in 0..NN_OUTPUT_SIZE {
        // Track highest probability overall.
        if nn.outputs[i] > highest_prob {
            highest_prob = nn.outputs[i];
            highest_prob_idx = i as i8;
        }

        // Track best legal move.
        if state.board[i] == '.' && (best_move == -1 || nn.outputs[i] > best_legal_prob) {
            best_move = i as i8;
            best_legal_prob = nn.outputs[i];
        }
    }

    if display_move {
        print!("Neural network move probabilities:\n");
        let mut pos;
        for row in 0..ROWS_SIZE {
            for col in 0..ROWS_SIZE {
                pos = row * ROWS_SIZE + col;

                // Print probability as percentage.
                print!("{:.1}%", nn.outputs[pos] * 100.0);

                // Add markers.
                if (pos as i8) == highest_prob_idx {
                    print!("*"); // Highest probability overall.
                }
                if (pos as i8) == best_move {
                    print!("#"); // Selected move (highest valid probability).
                }
                print!(" ");
            }
            println!("");
        }
    }

    return best_move as usize;
}

/// Get a random valid move, this is used for training
/// against a random opponent. Note: this function will loop forever
/// if the board is full, but here we want simple code.
fn play_random_move(state: &GameState) -> usize {
    let mut rng = rand::rng();
    let mut move_random: usize;
    loop {
        move_random = rng.random_range(0..BOARD_SIZE);
        if state.board[move_random] == '.' {
            return move_random;
        }
    }
}

/// is_game_over checks if a specific state is game over for the tic tac toe game.
/// We have a game over in the following scenarios:
/// 1) All the elemens in a row are "O" (or "X").
/// 2) All the elements in a column are "O" (or "X").
/// 3) All the elements in a diagonal are "O" (or "X").
fn is_game_over(state: &GameState, winner: &mut char) -> bool {
    // Check the rows
    for i in 0..ROWS_SIZE {
        if state.board[i * ROWS_SIZE] == '.' {
            continue;
        }

        if state.board[i * ROWS_SIZE] == state.board[(i * ROWS_SIZE) + 1]
            && state.board[(i * ROWS_SIZE) + 1] == state.board[(i * ROWS_SIZE) + 2]
            && state.board[(i * ROWS_SIZE) + 2] == state.board[(i * ROWS_SIZE) + 3]
        {
            *winner = state.board[i * ROWS_SIZE];
            return true;
        }
    }

    // check the columns
    for i in 0..ROWS_SIZE {
        if state.board[i] == '.' {
            continue;
        }

        if state.board[i] == state.board[i + ROWS_SIZE]
            && state.board[i + ROWS_SIZE] == state.board[i + ROWS_SIZE * 2]
            && state.board[i + ROWS_SIZE * 2] == state.board[i + ROWS_SIZE * 3]
        {
            *winner = state.board[i];
            return true;
        }
    }

    // check the diagonals
    if state.board[0] != '.'
        && state.board[0] == state.board[5]
        && state.board[5] == state.board[10]
        && state.board[10] == state.board[15]
    {
        *winner = state.board[0];
        return true;
    }

    if state.board[3] != '.'
        && state.board[3] == state.board[6]
        && state.board[6] == state.board[9]
        && state.board[9] == state.board[12]
    {
        *winner = state.board[3];
        return true;
    }

    // Check for tie (no free tiles left).
    let mut empty_tiles: u8 = 0;
    for i in 0..BOARD_SIZE {
        if state.board[i] == '.' {
            empty_tiles += 1
        }
    }

    if empty_tiles == 0 {
        *winner = 'T'; // Tie
        return true;
    }

    return false;
}

/* Play a game against random moves and learn from it.
 *
 * This is a very simple Montecarlo Method applied to reinforcement
 * learning:
 *
 * 1. We play a complete random game (episode).
 * 2. We determine the reward based on the outcome of the game.
 * 3. We update the neural network in order to maximize future rewards.
 *
 * LEARNING OPPORTUNITY: while the code uses some Montecarlo-alike
 * technique, important results were recently obtained using
 * Montecarlo Tree Search (MCTS), where a tree structure represents
 * potential future game states that are explored according to
 * some selection: you may want to learn about it. */
fn play_random_game(nn: &mut NeuralNetwork) -> char {
    let mut state: GameState = GameState {
        board: ['.'; BOARD_SIZE],
        current_player: false,
    };

    let mut winner: char = '.';
    let mut move_history: [usize; BOARD_SIZE] = [0; BOARD_SIZE];
    let mut num_moves: usize = 0;
    let mut h_move: usize;
    while !is_game_over(&state, &mut winner) {
        if state.current_player {
            // Neural Network Move
            h_move = play_computer_move(&state, nn, false);
            state.board[h_move] = 'O';
        } else {
            // human move -> get a random valid move
            h_move = play_random_move(&state);
            state.board[h_move] = 'X';
        }

        // Store the move: we need the moves sequence
        // during the learning stage.
        move_history[num_moves] = h_move;
        num_moves += 1;

        // switch player
        state.current_player = !state.current_player
    }

    learn_from_game(nn, &move_history, &num_moves, true, winner);

    return winner;
}

/// Train the neural network based on game outcome.
///
/// The move_history is just an integer array with the index of all the
/// moves. This function is designed so that you can specify if the
/// game was started by the move by the NN or human, but actually the
/// code always let the human move first.
fn learn_from_game(
    nn: &mut NeuralNetwork,
    move_history: &[usize; BOARD_SIZE],
    num_moves: &usize,
    nn_moves_even: bool,
    winner: char,
) {
    let reward;
    let mut nn_symbol: char = 'X';
    if nn_moves_even {
        // NN started first
        nn_symbol = 'O';
    }

    if winner == 'T' {
        reward = 0.3; // Small reward for draw
    } else if winner == nn_symbol {
        reward = 1.0; // Large reward for win
    } else {
        reward = -1.0; // Modified: Less severe negative reward (-1.0 instead of -2.0)
    }

    let mut target_probs = [0.0; NN_OUTPUT_SIZE];

    // Process each move the nn made.
    for move_idx in 0..*num_moves {
        // Skip moves that were not made by the NN
        if (nn_moves_even && move_idx % 2 != 1) || (!nn_moves_even && move_idx % 2 != 0) {
            continue;
        }

        // Recreate board state BEFORE this move was made.
        let mut state = GameState {
            board: ['.'; BOARD_SIZE], // We use '.' to identify an empty cell.
            current_player: false,
        };
        let mut symbol;
        for i in 0..move_idx {
            if i % 2 == 0 {
                symbol = 'X'
            } else {
                symbol = 'O'
            }

            state.board[move_history[i]] = symbol;
        }

        // Convert board to inputs and do forward pass.
        let mut inputs = [0.0; NN_INPUT_SIZE];
        board_to_inputs(&state, &mut inputs);
        forward_pass(nn, inputs);

        let current_move = move_history[move_idx];
        let move_importance = 0.5 + 0.5 * (move_idx as f64 / *num_moves as f64);
        let scaled_reward = reward * move_importance; // Removed the f32 cast to keep as f64

        for i in 0..NN_OUTPUT_SIZE {
            target_probs[i] = 0.0;
        }

        // Set the target for the chosen move based on reward:
        if scaled_reward >= 0.0 {
            // For positive reward, set probability of the chosen move to
            // 1, with all the rest set to 0.
            target_probs[current_move] = 1.0;
        } else {
            // For negative reward, distribute probability to OTHER
            // valid moves, which is conceptually the same as discouraging
            // the move that we want to discourage.
            let mut valid_moves_count = 0;
            for i in 0..BOARD_SIZE {
                if state.board[i] == '.' && i != current_move {
                    valid_moves_count += 1;
                }
            }

            if valid_moves_count > 0 {
                let other_prob = 1.0 / valid_moves_count as f64;
                for i in 0..BOARD_SIZE {
                    if state.board[i] == '.' && i != current_move {
                        target_probs[i] = other_prob;
                    }
                }
            } else {
                // If there are no other valid moves, just discourage the current move
                target_probs[current_move] = 0.0;
            }
        }

        backprop(nn, &target_probs, LEARNING_RATE, scaled_reward);
    }
}

/// Backpropagation function.
/// The only difference here from vanilla backprop is that we have
/// a 'reward_scaling' argument that makes the output error more/less
/// dramatic, so that we can adjust the weights proportionally to the
/// reward we want to provide. */
fn backprop(
    nn: &mut NeuralNetwork,
    target_probs: &[f64; NN_OUTPUT_SIZE],
    learning_rate: f64,
    reward_scaling: f64, // Changed from f32 to f64 for consistency
) {
    let mut output_deltas: [f64; NN_OUTPUT_SIZE] = [0.0; NN_OUTPUT_SIZE];
    let mut hidden2_deltas: [f64; NN_HIDDEN_SIZE] = [0.0; NN_HIDDEN_SIZE];
    let mut hidden_deltas: [f64; NN_HIDDEN_SIZE] = [0.0; NN_HIDDEN_SIZE];

    // === STEP 1: Compute deltas ===

    // Calculate output layer deltas:
    // Note what's going on here: we are technically using softmax
    // as output function and cross entropy as loss, but we never use
    // cross entropy in practice since we check the progresses in terms
    // of winning the game.
    //
    // Still calculating the deltas in the output as:
    //
    //      output[i] - target[i]
    //
    // Is exactly what happens if you derivate the deltas with
    // softmax and cross entropy.
    //
    // LEARNING OPPORTUNITY: This is a well established and fundamental
    // result in neural networks, you may want to read more about it.
    for i in 0..NN_OUTPUT_SIZE {
        output_deltas[i] = (nn.outputs[i] - target_probs[i]) * libm::fabs(reward_scaling);
    }

    // Backpropagate error to hidden layer 2.
    for i in 0..NN_HIDDEN_SIZE {
        let mut error: f64 = 0.0;
        for j in 0..NN_OUTPUT_SIZE {
            error += output_deltas[j] * nn.weights_ho[i * NN_OUTPUT_SIZE + j];
        }
        hidden2_deltas[i] = error * relu_derivative(nn.hidden2[i]);
    }

    // Hidden layer 1 deltas
    for i in 0..NN_HIDDEN_SIZE {
        let mut error: f64 = 0.0;
        for j in 0..NN_HIDDEN_SIZE {
            error += hidden2_deltas[j] * nn.weights_hh[i * NN_HIDDEN_SIZE + j];
        }
        hidden_deltas[i] = error * relu_derivative(nn.hidden[i]);
    }

    // === STEP 2: Weights updating ===

    // Output layer weights and biases.
    for i in 0..NN_HIDDEN_SIZE {
        for j in 0..NN_OUTPUT_SIZE {
            nn.weights_ho[i * NN_OUTPUT_SIZE + j] -=
                learning_rate * output_deltas[j] * nn.hidden2[i];
        }
    }

    for j in 0..NN_OUTPUT_SIZE {
        nn.biases_o[j] -= learning_rate * output_deltas[j];
    }

    // Hidden layer 2 weights and biases.
    for i in 0..NN_HIDDEN_SIZE {
        for j in 0..NN_HIDDEN_SIZE {
            nn.weights_hh[i * NN_HIDDEN_SIZE + j] -=
                learning_rate * hidden2_deltas[j] * nn.hidden[i];
        }
    }

    for j in 0..NN_HIDDEN_SIZE {
        nn.biases_h2[j] -= learning_rate * hidden2_deltas[j];
    }

    // Hidden layer 1 weights and biases.
    for i in 0..NN_INPUT_SIZE {
        for j in 0..NN_HIDDEN_SIZE {
            nn.weights_ih[i * NN_HIDDEN_SIZE + j] -=
                learning_rate * hidden_deltas[j] * nn.inputs[i];
        }
    }

    for j in 0..NN_HIDDEN_SIZE {
        nn.biases_h[j] -= learning_rate * hidden_deltas[j];
    }
}

/// Train the neural network against random moves.
pub fn train_against_random(nn: &mut NeuralNetwork, num_games: u32) {
    println!(
        "\nTraining neural network with two hidden layer against {} random games...",
        num_games
    );

    let mut wins: u32 = 0;
    let mut losses: u32 = 0;
    let mut ties: u32 = 0;
    let mut played_games = 0;

    for i in 0..num_games {
        let winner = play_random_game(nn);
        played_games += 1;

        if winner == 'O' {
            wins += 1;
        } else if winner == 'X' {
            losses += 1;
        } else {
            ties += 1;
        }

        // Print training stats on the console.
        if ((i + 1) % 1000) == 0 {
            println!(
                "Games: {}, Wins: {} ({:.1}%), Losses: {} ({:.1})%, Ties: {} ({:.1}%)",
                i + 1,
                wins,
                (wins as f32 / played_games as f32) * 100.0,
                losses,
                (losses as f32 / played_games as f32) * 100.0,
                ties,
                (ties as f32 / played_games as f32) * 100.0
            );

            played_games = 0;
            wins = 0;
            losses = 0;
            ties = 0;
        }
    }
}

/* Show board on screen in ASCII */
fn display_board(state: &GameState) {
    println!("");
    for row in 0..ROWS_SIZE {
        // Display the board symbols.
        print!(
            "{}{}{}{} ",
            state.board[row * ROWS_SIZE],
            state.board[row * ROWS_SIZE + 1],
            state.board[row * ROWS_SIZE + 2],
            state.board[row * ROWS_SIZE + 3],
        );

        // Display the position numbers for this row, for the poor human.
        println!(
            "{} {} {} {}",
            row * ROWS_SIZE,
            row * ROWS_SIZE + 1,
            row * ROWS_SIZE + 2,
            row * ROWS_SIZE + 3
        );
    }
    println!("");
}

fn play_game(nn: &mut NeuralNetwork) {
    println!("\nWelcome to Tic Tac Toe! You are X, the computer is O.");
    println!(
        "Enter positions as numbers from 0 to {} (see picture).",
        BOARD_SIZE - 1
    );

    let mut state: GameState = GameState {
        board: ['.'; BOARD_SIZE],
        current_player: false,
    };

    let stdin = io::stdin();
    let mut winner: char = '.';
    let mut human_move: String;
    while !is_game_over(&state, &mut winner) {
        display_board(&state);
        if state.current_player == false {
            // Human move
            human_move = String::new();
            println!("Your move (0-{}): ", BOARD_SIZE - 1);
            stdin.lock().read_line(&mut human_move).unwrap();
            let input = human_move.trim().parse::<usize>().unwrap();

            if input > BOARD_SIZE - 1 || state.board[input] != '.' {
                println!("Invalid move! Try again.");
                continue;
            }

            state.board[input] = 'X';
        } else {
            println!("Computer's move: \n");
            let h_move = play_computer_move(&state, nn, true);
            state.board[h_move] = 'O';
            println!("\nComputer's placed O at position: {}:", h_move);
        }
        state.current_player = !state.current_player;
    }

    display_board(&state);
    if winner == 'X' {
        println!("You win!");
    } else if winner == 'O' {
        println!("Computer wins!");
    } else {
        println!("It's a tie!");
    }
}

fn main() {
    let mut random_games: u32 = 150000;
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        random_games = args[1].parse().unwrap();
    }

    // Init NN.
    let mut nn = init_nn();
    // Train the NN.
    if random_games > 0 {
        train_against_random(&mut nn, random_games);
    }

    let stdin = io::stdin();
    let mut play_again: String;
    loop {
        play_again = String::new();
        play_game(&mut nn);

        println!("\nPlay again? (y/n): ");
        stdin.lock().read_line(&mut play_again).unwrap();
        let input = play_again.trim();
        if input != "y" && input != "Y" {
            break;
        }
    }
}
