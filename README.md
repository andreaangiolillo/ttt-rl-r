Learning Project: A Simple Neural Network to play tic tac toe

The repository has three branches:
- [3x3](https://github.com/andreaangiolillo/ttt-rl-r/edit/3x3): defines a simple neural network with one hidden layer to play tic tac toe against a human in a 3x3 board.
- [4x4](https://github.com/andreaangiolillo/ttt-rl-r/edit/4x4): defines a simple neural network with one hidden layer to play tic tac toe against a human in a 4x4 board.
- [4x4_one_layer_against_two_layers](https://github.com/andreaangiolillo/ttt-rl-r/edit/4x4_one_layer_against_two_layers): defines a simple neural network with one hidden layer and another neural network with two hidden layes and let play against each other in a 4x4 board.

# How to run the code

### [Install Rust and Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)
```bash
curl https://sh.rustup.rs -sSf | sh
```
### Build the code
```bash
cargo build --release
```
### Run the binary
```bash
./target/release/ttt
```
