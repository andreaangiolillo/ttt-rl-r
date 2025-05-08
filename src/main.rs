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

fn play_random_game() -> char {
    return 'O';
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
        let winner = play_random_game();
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
    let random_games: u32 = 150000;

    // Init Game State.
    let game_state = GameState {
        board: ['.'; BOARD_SIZE], // We use '.' to identify an empty cell.
        current_player: false,    // The humam plays the first move
    };

    // Init NN.
    let mut nn = init_nn();

    // Train the NN.
    if random_games > 0 {
        train_against_random(&nn, random_games);
    }

    let mut rng = rand::rng();

    // Print text to the console.
    // for i in 1..100{
    //     println!("Hello World! {}", rng.random::<f64>());
    // }
}
