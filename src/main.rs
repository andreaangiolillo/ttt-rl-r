use rand::Rng;

/// Neural network parameters.
const NN_INPUT_SIZE: usize = 18;
const NN_HIDDEN_SIZE: usize = 100;
const NN_OUTPUT_SIZE: usize = 9;
const BOARD_SIZE: usize = 9;
const LEARNING_RATE: f32 = 0.01;

/// Game board structure.
struct GameState {
    board: [char; BOARD_SIZE],
    current_player: bool, // False for human (X), True for computer (O)
}

/// Neural network structure. For simplicity we have just
/// one hidden layer and fixed sizes (see defines above).
/// However for this problem going deeper than one hidden layer
/// is useless.
struct NeuralNetwork {
    weights_ih: [f64; NN_INPUT_SIZE * NN_HIDDEN_SIZE],
    weights_ho: [f64; NN_HIDDEN_SIZE * NN_OUTPUT_SIZE],
    biases_h: [f64; NN_HIDDEN_SIZE],
    biases_o: [f64; NN_OUTPUT_SIZE],

    inputs: [f64; NN_INPUT_SIZE],
    hidden: [f64; NN_HIDDEN_SIZE],
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
    return rng.random::<f64>();
}

/// Initialize the NN
fn init_nn() -> NeuralNetwork {
    let mut nn = NeuralNetwork {
        weights_ih: [0.0; NN_INPUT_SIZE * NN_HIDDEN_SIZE],
        weights_ho: [0.0; NN_HIDDEN_SIZE * NN_OUTPUT_SIZE],
        biases_h: [0.0; NN_HIDDEN_SIZE],
        biases_o: [0.0; NN_OUTPUT_SIZE],
        inputs: [0.0; NN_INPUT_SIZE],
        hidden: [0.0; NN_HIDDEN_SIZE],
        raw_logits: [0.0; NN_OUTPUT_SIZE],
        outputs: [0.0; NN_OUTPUT_SIZE],
    };

    for i in 0..(NN_INPUT_SIZE * NN_HIDDEN_SIZE) {
        nn.weights_ih[i] = random_weight();
    }

    for i in 0..(NN_HIDDEN_SIZE * NN_OUTPUT_SIZE) {
        nn.weights_ho[i] = random_weight();
    }

    for i in 0..NN_HIDDEN_SIZE {
        nn.biases_h[i] = random_weight();
    }

    for i in 0..NN_OUTPUT_SIZE {
        nn.biases_o[i] = random_weight()
    }

    return nn;
}

/// Convert board state to neural network inputs. Note that we use
/// a peculiar encoding I descrived here:
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
    for i in 0..9 {
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

fn forward_pass(nn: &NeuralNetwork, inputs:[f64; NN_INPUT_SIZE]) {

}

/// Get the best move computer move using the neural network.
/// Note that there is no complex sampling at all, we just get
/// the output with the highest value that has an empty tile.
fn play_computer_move(state: &GameState, nn: &NeuralNetwork, display_move: bool) -> usize {
    let mut inputs = [0.0; NN_INPUT_SIZE];

    board_to_inputs(state, &mut inputs);
    // println!("\ninputs:");
    // for i in 0..18 {
    //     if (i % 2) == 0 {
    //         println!("{} {}", inputs[i], inputs[i+1] )
    //     }
    // }

    forward_pass(nn, inputs);

    let mut move_random: usize = 0;
    // let mut rng = rand::rng();
    // let mut move_random: usize = 0;
    // while true {
    //     move_random = (rng.random::<u8>() % 9) as usize;
    //     if state.board[move_random] == '.' {
    //         return move_random;
    //     }
    // }
    // return move_random;
    return move_random;
}

/// Get a random valid move, this is used for training
/// against a random opponent. Note: this function will loop forever
/// if the board is full, but here we want simple code.
fn play_random_move(state: &GameState) -> usize {
    let mut rng = rand::rng();
    let mut move_random: usize = 0;
    while true {
        move_random = (rng.random::<u8>() % 9) as usize;
        if state.board[move_random] == '.' {
            return move_random;
        }
    }
    return move_random;
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
fn play_random_game(nn: &NeuralNetwork, move_history: [char; 9]) -> char {
    let mut state: GameState = GameState {
        board: ['.'; BOARD_SIZE],
        current_player: false,
    };

    let mut winner: char = '.';
    let mut num_moves: u32 = 0;
    let mut move_round: u16;
    while !is_game_over(&state, &mut winner) {
        if state.current_player {
            // Neural Network Move
            let h_move = play_computer_move(&state, nn, true);
            println!("NN Move!!!: {}", h_move);
            state.board[h_move] = 'O';
        } else {
            // human move -> get a random valid move
            let h_move = play_random_move(&state);
            println!("Human Move!!! (random): {}", h_move);
            state.board[h_move] = 'X';
        }

        println!("\nEND TURN! Board:");
        println!(
            "{} {} {}\n{} {} {}\n{} {} {}",
            state.board[0],
            state.board[1],
            state.board[2],
            state.board[3],
            state.board[4],
            state.board[5],
            state.board[6],
            state.board[7],
            state.board[8]
        );
        println!("\n");
        state.current_player = !state.current_player
    }

    return 'O';
}

/// is_game_over checks if a specific state is game over for the tic tac toe game.
/// We have a game over in the following scenarios:
/// 1) All the elemens in a row are "O" (or "X").
/// 2) All the elements in a column are "O" (or "X").
/// 3) All the elements in a diagonal are "O" (or "X").
fn is_game_over(state: &GameState, winner: &mut char) -> bool {
    // Check the rows
    for i in 0..3 {
        if state.board[i * 3] == '.' {
            continue;
        }

        if state.board[i * 3] == state.board[(i * 3) + 1]
            && state.board[(i * 3) + 1] == state.board[(i * 3) + 2]
        {
            *winner = state.board[i * 3];
            return true;
        }
    }

    // check the columns
    for i in 0..3 {
        if state.board[i] == '.' {
            continue;
        }

        if state.board[i] == state.board[i + 3] && state.board[i + 3] == state.board[i + 6] {
            *winner = state.board[i];
            return true;
        }
    }

    // check the diagonals
    if state.board[4] == '.' {
        return false;
    }

    if state.board[0] == state.board[4] && state.board[4] == state.board[8] {
        *winner = state.board[0];
        return true;
    }

    if state.board[2] == state.board[4] && state.board[4] == state.board[6] {
        *winner = state.board[2];
        return true;
    }

    // Check for tie (no free tiles left).
    let mut empty_tiles: u8 = 0;
    for i in 0..9 {
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

/// Train the neural network against random moves.
fn train_against_random(nn: &NeuralNetwork, num_games: u32) {
    println!(
        "\nTraining neural network against {} random games...",
        num_games
    );

    let mut move_history: [char; 9] = ['.'; 9];
    let mut wins: u32 = 0;
    let mut losses: u32 = 0;
    let mut ties: u32 = 0;

    let mut played_games = 0;
    for i in 0..num_games {
        let winner = play_random_game(nn, move_history);
        played_games += 1;

        if winner == 'O' {
            wins += 1;
        } else if winner == 'X' {
            losses += 1;
        } else {
            ties += 1;
        }

        // Print training stats on the console.
        if ((i + 1) % 10000) == 0 {
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

fn main() {
    let random_games: u32 = 1;

    // Init Game State.
    let game_state = GameState {
        board: ['.'; BOARD_SIZE], // We use '.' to identify an empty cell.
        current_player: false,    // The humam plays the first move
    };

    // Init NN.
    let nn = init_nn();

    // Train the NN.
    if random_games > 0 {
        train_against_random(&nn, random_games);
    }

    // let mut rng = rand::rng();
    // Print text to the console.
    // for i in 1..100{
    //     println!("Hello World! {}", rng.random::<f64>());
    // }
}
