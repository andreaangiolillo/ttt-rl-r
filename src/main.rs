mod nn_one_hidden_layer;
mod nn_two_hidden_layers;
use std::env;

fn play_game(
    random_games: u32,
    one_layer_nn: &mut nn_one_hidden_layer::NeuralNetwork,
    two_layers_nn: &mut nn_two_hidden_layers::NeuralNetwork,
) {
    println!(
        "\nWelcome to Tic Tac Toe! Neural Network with one layer is X, and the Neural Network with two layers is O."
    );
    println!(
        "\nWe will play {} games and see which one is better!",
        random_games
    );

    let mut state = nn_one_hidden_layer::GameState {
        board: ['.'; BOARD_SIZE],
        current_player: false,
    };

    let mut winner: char = '.';
    let mut nn1_wins: i64 = 0;
    let mut nn2_wins: i64 = 0;
    let mut ties: i64 = 0;
    let mut n_games: i64 = 0;
    while n_games < random_games.into() {
        while !nn_one_hidden_layer::is_game_over(&state, &mut winner) {
            if state.current_player == false {
                // NN 1 layer move
                let h_move = nn_one_hidden_layer::play_computer_move(&state, one_layer_nn, false);
                state.board[h_move] = 'X';
            } else {
                let h_move = nn_two_hidden_layers::play_computer_move_nn(&state, two_layers_nn, false);
                state.board[h_move] = 'O';
            }
            state.current_player = !state.current_player;
        }

        // display_board(&state);
        if winner == 'X' {
            nn1_wins += 1;
        } else if winner == 'O' {
            nn2_wins += 1;
        } else {
            ties += 1
        }

        n_games += 1;
    }

    println!(
        "\nAfter {} games, Neural Network with one layer won: {}, Neural Network with two layers won: {}, ties: {}",
        n_games, nn1_wins, nn2_wins, ties
    );
}

fn main() {
    let mut random_games: u32 = 250000;
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        random_games = args[1].parse().unwrap();
    }

    // Init NN.
    let mut nn_one_layer = nn_one_hidden_layer::init_nn();
    let mut nn_two_layers = nn_two_hidden_layers::init_nn();

    // Train the NN.
    if random_games > 0 {
        nn_one_hidden_layer::train_against_random(&mut nn_one_layer, random_games);
        nn_two_hidden_layers::train_against_random(&mut nn_two_layers, random_games);
    }
    play_game(random_games, &mut nn_one_layer, &mut nn_two_layers);
}
