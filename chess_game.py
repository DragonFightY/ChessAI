import io
import json
import logging
import os
import random
import threading
import time
from functools import lru_cache
from typing import Any, List

import chess
import chess.svg
import cairosvg
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

# CONSTANTS / CONFIGURATION
LOG_FILE = 'chess_game.log'
METRICS_FILE = 'chess_metrics.json'
TRAINING_DATA_FILE = 'training_data.pth'

# Standard logging configuration
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Custom logging handler to notify errors via the GUI.
class GUIErrorHandler(logging.Handler):
    def __init__(self, tk_root: tk.Tk):
        super().__init__(level=logging.ERROR)
        self.root = tk_root

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Schedule a messagebox on the GUI thread.
            self.root.after(0, lambda: messagebox.showerror("Error", msg))
        except Exception:
            pass


class ChessMetrics:
    """
    Track and store chess game metrics including win/loss statistics,
    game history, and performance trends.
    """
    def __init__(self) -> None:
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.total_draws: int = 0
        self.game_history: List[Any] = []
        self.cumulative_wins: List[int] = []
        self.training_games_played: int = 0
        self.metrics_file: str = METRICS_FILE
        self.load_metrics()
    
    def load_metrics(self) -> None:
        """Load saved metrics from file"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.total_wins = data.get('wins', 0)
                    self.total_losses = data.get('losses', 0)
                    self.total_draws = data.get('draws', 0)
                    self.game_history = data.get('history', [])
                    self.cumulative_wins = data.get('cumulative_wins', [])
                    self.training_games_played = data.get('training_games', 0)
            except (json.JSONDecodeError, OSError) as e:
                logging.error(f"Error loading metrics from {self.metrics_file}: {e}")
    
    def save_metrics(self) -> None:
        """Save current metrics to file"""
        try:
            data = {
                'wins': self.total_wins,
                'losses': self.total_losses,
                'draws': self.total_draws,
                'history': self.game_history,
                'cumulative_wins': self.cumulative_wins,
                'training_games': self.training_games_played
            }
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f)
        except OSError as e:
            logging.error(f"Error saving metrics to {self.metrics_file}: {e}")

    def update_metrics(self, result: str) -> None:
        """Update metrics with game result and save"""
        self.game_history.append(result)
        if result == 'win':
            self.total_wins += 1
        elif result == 'loss':
            self.total_losses += 1
        else:
            self.total_draws += 1
        self.cumulative_wins.append(self.total_wins)
        self.training_games_played += 1
        self.save_metrics()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EnhancedChessNet(nn.Module):
    def __init__(self) -> None:
        super(EnhancedChessNet, self).__init__()
        # Input layers
        self.conv1 = nn.Conv2d(6, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(64)
        self.policy_fc = nn.Linear(64 * 64, 1968)  # All possible moves
        
        # Value head
        self.value_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 64, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Initial convolution
        out = self.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            out = block(out)
        
        # Policy head
        policy = self.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(-1, 64 * 64)
        policy = self.policy_fc(policy)
        policy = torch.softmax(policy, dim=1)
        
        # Value head
        value = self.relu(self.value_bn(self.value_conv(out)))
        value = value.view(-1, 32 * 64)
        value = self.relu(self.value_fc1(value))
        value = self.dropout(value)
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


class ChessAI:
    """
    Enhanced Chess AI with improved evaluation and learning capabilities.
    Includes position evaluation, move ordering, and adaptive learning rate.
    """
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EnhancedChessNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        self.game_history: List[str] = []
        self.training_games: int = 0
        # Lock to synchronize access to game_history
        self.history_lock = threading.Lock()
        self.metrics = ChessMetrics()
        self.load_training_state()
        self.model.eval()

    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Convert chess board state to tensor format"""
        pieces = 'pnbrqk'
        tensor = np.zeros((6, 8, 8), dtype=np.float32)
        
        for i in range(64):
            piece = board.piece_at(i)
            if piece is not None:
                piece_type = pieces.index(piece.symbol().lower())
                rank, file = i // 8, i % 8
                tensor[piece_type][rank][file] = 1 if piece.color else -1
        
        return torch.FloatTensor(tensor).unsqueeze(0).to(self.device)

    @lru_cache(maxsize=1000000)
    def evaluate_position_cached(self, board_fen: str) -> float:
        """
        Evaluate the board and return a single float value.
        Uses the neural network's value head and adds a mobility bonus.
        """
        board = chess.Board(board_fen)

        # Terminal positions return plain floats.
        if board.is_checkmate():
            return -1000 if board.turn else 1000
        if board.is_stalemate():
            return 0

        # Evaluate using the network & extract a scalar
        with torch.no_grad():
            state = self.board_to_tensor(board)
            policy, value = self.model(state)
            eval_score = value.item()  # Returns a float

        # Add bonus for mobility (float arithmetic)
        mobility_score = len(list(board.legal_moves)) * 0.01
        eval_score += mobility_score if board.turn else -mobility_score

        return eval_score

    def train_on_game(self, game_history: List[str], final_result: str) -> float:
        """Train neural network on completed game and return average loss"""
        self.model.train()
        running_loss = 0.0

        # Convert result to target value
        if final_result == 'win':
            target_value = 1.0
        elif final_result == 'loss':
            target_value = -1.0
        else:
            target_value = 0.0

        gamma = 0.99  # Discount factor for TD learning

        for i in range(len(game_history) - 1):
            current_board = chess.Board(game_history[i])
            next_board = chess.Board(game_history[i + 1])

            current_tensor = self.board_to_tensor(current_board)
            next_tensor = self.board_to_tensor(next_board)

            with torch.no_grad():
                _, next_value = self.model(next_tensor)
            _, current_value = self.model(current_tensor)

            # TD target: if last move, use final result; otherwise apply discount factor
            if i == len(game_history) - 2:
                target = torch.tensor([[target_value]], device=self.device)
            else:
                target = gamma * next_value

            loss = nn.MSELoss()(current_value, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / (len(game_history) - 1)
        self.scheduler.step(avg_loss)
        self.training_games += 1
        return avg_loss

    def get_best_move(self, board: chess.Board, max_depth: int = 3) -> chess.Move:
        """
        Find the best move using iterative deepening and alpha-beta pruning.
        """
        best_move = None
        try:
            ordered_moves = self.order_moves(board)
            
            for depth in range(1, max_depth + 1):
                current_best_move = None
                current_best_value = float('-inf')
                depth_alpha = float('-inf')
                depth_beta = float('inf')
                
                for move in ordered_moves:
                    board.push(move)
                    value = -self.minimax(board, depth-1, -depth_beta, -depth_alpha)
                    board.pop()
                    
                    if value > current_best_value:
                        current_best_value = value
                        current_best_move = move
                    depth_alpha = max(depth_alpha, value)
                    
                best_move = current_best_move
                # Update move ordering with best move at the front
                ordered_moves = [best_move] + [m for m in ordered_moves if m != best_move]
            
            return best_move
        except Exception as e:
            logging.error(f"Error in get_best_move: {e}")
            return random.choice(list(board.legal_moves))

    def order_moves(self, board: chess.Board) -> List[chess.Move]:
        """Order moves to improve alpha-beta pruning efficiency."""
        moves = list(board.legal_moves)
        scored_moves = []
        
        for move in moves:
            score = 0
            # Prioritize captures using MVV-LVA
            if board.is_capture(move):
                victim_piece = board.piece_at(move.to_square)
                attacker_piece = board.piece_at(move.from_square)
                if victim_piece and attacker_piece:
                    piece_values = {
                        chess.PAWN: 1, chess.KNIGHT: 3,
                        chess.BISHOP: 3, chess.ROOK: 5,
                        chess.QUEEN: 9, chess.KING: 0
                    }
                    score = 10 * piece_values[victim_piece.piece_type] - piece_values[attacker_piece.piece_type]
            
            # Prioritize promotions
            if move.promotion:
                score += 900
            
            # Prioritize center control
            to_rank, to_file = chess.square_rank(move.to_square), chess.square_file(move.to_square)
            if 2 <= to_rank <= 5 and 2 <= to_file <= 5:
                score += 1
            
            # Temporarily push move to check for check condition
            board.push(move)
            if board.is_check():
                score += 5
            board.pop()
            
            scored_moves.append((score, move))
        
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        """Minimax algorithm with alpha-beta pruning."""
        if depth == 0 or board.is_game_over():
            return self.evaluate_position_cached(board.fen())
        
        ordered_moves = self.order_moves(board)
        best_value = float('-inf')
        
        for move in ordered_moves:
            board.push(move)
            value = -self.minimax(board, depth-1, -beta, -alpha)
            board.pop()
            
            best_value = max(best_value, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        
        return best_value

    def load_training_state(self) -> None:
        """
        Load training data (model, optimizer, training_games, and game_history)
        from file.
        """
        if os.path.exists(TRAINING_DATA_FILE):
            try:
                training_data = torch.load(TRAINING_DATA_FILE, map_location=self.device)
                self.model.load_state_dict(training_data['model_state'])
                self.optimizer.load_state_dict(training_data['optimizer_state'])
                self.training_games = training_data.get('training_games', 0)
                self.game_history = training_data.get('game_history', [])
                logging.info("Loaded training data from file")
            except Exception as e:
                logging.warning(f"Could not load training data: {e}")
        else:
            logging.info("No training data file found. Starting fresh.")

    def save_training_state(self, force: bool = False) -> None:
        """
        Save training data (model, optimizer, training_games, and game_history)
        to file.
        """
        try:
            if self.training_games > 0 or force:
                training_data = {
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'training_games': self.training_games,
                    'game_history': self.game_history
                }
                torch.save(training_data, TRAINING_DATA_FILE)
                logging.info("Saved training data")
        except Exception as e:
            logging.error(f"Error saving training data: {e}")

    def store_move(self, board: chess.Board) -> None:
        """Store the current board position for training."""
        try:
            with self.history_lock:
                self.game_history.append(board.fen())
        except Exception as e:
            logging.error(f"Error storing move: {e}")


class ChessGUI:
    """
    Enhanced GUI for chess game with real-time training visualization,
    custom error logging, asynchronous training updates,
    and improved training state handling.
    """
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Advanced Chess AI Interface")
        
        self.board = chess.Board()
        self.selected_square = None
        self.training_in_progress = False
        self.game_active = True
        self.status_var = tk.StringVar(value="White to move")
        self.player_color = chess.WHITE
        
        self.main_container = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        self.left_panel = ttk.Frame(self.main_container)
        self.main_container.add(self.left_panel)
        
        self.right_panel = ttk.Frame(self.main_container)
        self.main_container.add(self.right_panel)
        
        try:
            self.chess_ai = ChessAI()
            self.has_ai = True
        except Exception as e:
            self.has_ai = False
            messagebox.showwarning("Warning", f"AI initialization failed: {str(e)}")
        
        self.setup_board_panel()
        self.setup_control_panel()
        self.setup_metrics_panel()
        self.setup_training_panel()
        
        self.update_metrics_display()
        self.update_board()

    def setup_board_panel(self) -> None:
        """Setup the main chess board display."""
        self.canvas_size = 400
        self.square_size = self.canvas_size // 8
        self.canvas = tk.Canvas(self.left_panel, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind('<Button-1>', self.on_square_click)

    def setup_control_panel(self) -> None:
        """Setup game control buttons and options."""
        control_frame = ttk.LabelFrame(self.right_panel, text="Game Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="New Game", command=self.new_game).pack(pady=5)
        
        self.color_var = tk.StringVar(value="white")
        ttk.Label(control_frame, text="Play as:").pack(pady=5)
        ttk.Radiobutton(control_frame, text="White", variable=self.color_var, 
                        value="white", command=self.change_color).pack()
        ttk.Radiobutton(control_frame, text="Black", variable=self.color_var, 
                        value="black", command=self.change_color).pack()

    def setup_metrics_panel(self) -> None:
        """Setup the metrics display panel with game statistics."""
        metrics_frame = ttk.LabelFrame(self.right_panel, text="Game Metrics")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.metrics_labels = {}
        metrics = [
            ("Wins", "wins"), ("Losses", "losses"), ("Draws", "draws"),
            ("Win Rate", "win_rate"), ("Games Played", "games_played")
        ]
        
        for i, (label, key) in enumerate(metrics):
            ttk.Label(metrics_frame, text=f"{label}:").grid(row=i, column=0, padx=5, pady=2, sticky="e")
            self.metrics_labels[key] = ttk.Label(metrics_frame, text="0")
            self.metrics_labels[key].grid(row=i, column=1, padx=5, pady=2, sticky="w")

    def setup_training_panel(self) -> None:
        """Setup the training control and visualization panel."""
        training_frame = ttk.LabelFrame(self.right_panel, text="Training Controls")
        training_frame.pack(fill=tk.X, padx=5, pady=5)
        
        settings_frame = ttk.Frame(training_frame)
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Number of games:").pack()
        self.num_games_var = tk.StringVar(value="100")
        ttk.Entry(settings_frame, textvariable=self.num_games_var).pack(pady=2)
        
        ttk.Label(settings_frame, text="Visualization Speed:").pack(pady=(5,0))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(settings_frame, from_=0.1, to=2.0,
                                variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_scale.pack(fill=tk.X, pady=2)
        
        self.progress_var = tk.StringVar(value="Training progress: 0%")
        self.progress_label = ttk.Label(training_frame, textvariable=self.progress_var)
        self.progress_label.pack(pady=2)
        
        self.progress_bar = ttk.Progressbar(training_frame, length=200, mode='determinate')
        self.progress_bar.pack(pady=5)
        
        self.viz_frame = ttk.LabelFrame(self.right_panel, text="Training Game")
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.training_canvas = tk.Canvas(self.viz_frame, width=200, height=200)
        self.training_canvas.pack(pady=5)
        
        button_frame = ttk.Frame(training_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.train_button = ttk.Button(button_frame, text="Start Training",
                                       command=self.start_self_play_training)
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Training",
                                      command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def self_play_training(self, num_games: int) -> None:
        """Execute self-play training games with visualization."""
        try:
            batch_size = 10  # save training state after every 10 games
            exploration_rate = 0.2  # 20% of the time choose a random move
            for game_num in range(num_games):
                if not self.training_in_progress:
                    break

                board = chess.Board()
                moves_without_capture = 0

                while not board.is_game_over() and moves_without_capture < 50:
                    if not self.training_in_progress:
                        break

                    # Update training display via UI thread
                    self.root.after(0, lambda b=board.copy(): self.update_training_display(b))
                    time.sleep(1 / self.speed_var.get())

                    # Add exploration: with probability exploration_rate choose a random move.
                    if random.random() < exploration_rate:
                        move = random.choice(list(board.legal_moves))
                    else:
                        move = self.chess_ai.get_best_move(board)

                    if move:
                        self.chess_ai.store_move(board)
                        is_capture = board.is_capture(move)
                        board.push(move)
                        moves_without_capture = 0 if is_capture else moves_without_capture + 1
                    else:
                        break

                result = 'draw'
                if board.is_checkmate():
                    result = 'win' if not board.turn else 'loss'

                self.root.after(0, self.chess_ai.metrics.update_metrics, result)
                progress = (game_num + 1) / num_games * 100
                self.root.after(0, self.update_training_progress, game_num + 1, progress)
                self.root.after(0, self.update_metrics_display)

                if self.training_in_progress:
                    with self.chess_ai.history_lock:
                        avg_loss = self.chess_ai.train_on_game(self.chess_ai.game_history, result)
                        self.chess_ai.game_history = []
                    logging.info(f"Game {game_num + 1} completed. Average loss: {avg_loss:.4f}")

                # Batch save training state every batch_size games
                if (game_num + 1) % batch_size == 0:
                    self.chess_ai.save_training_state(force=True)
        except Exception as e:
            logging.error(f"Training error: {e}")
            self.root.after(0, self.handle_training_error, str(e))
        finally:
            self.root.after(0, self.finish_training)

    def update_training_display(self, board: chess.Board) -> None:
        """Update the training visualization display."""
        try:
            svg_data = chess.svg.board(board, size=200)
            png_data = cairosvg.svg2png(bytestring=svg_data.encode())
            image = Image.open(io.BytesIO(png_data))
            self.training_photo = ImageTk.PhotoImage(image)
            
            self.training_canvas.delete("all")
            self.training_canvas.create_image(0, 0, image=self.training_photo, anchor="nw")
            self.root.update_idletasks()
        except Exception as e:
            logging.error(f"Error updating training display: {e}")

    def update_training_progress(self, games_completed: int, progress: float) -> None:
        """Update the training progress indicators."""
        self.progress_bar['value'] = games_completed
        self.progress_var.set(f"Training progress: {progress:.1f}%")

    def handle_training_error(self, error_message: str) -> None:
        """Handle and display training errors."""
        messagebox.showerror("Training Error", f"An error occurred during training: {error_message}")
        self.finish_training()

    def finish_training(self) -> None:
        """Clean up and reset after training completion."""
        self.training_in_progress = False
        self.game_active = True
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_var.set("Training complete!")
        self.chess_ai.save_training_state(force=True)

    def stop_training(self) -> None:
        """Stop the current training process."""
        self.training_in_progress = False
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.game_active = True

    def update_metrics_display(self) -> None:
        """Update all metrics displays with current statistics."""
        if not hasattr(self, 'chess_ai'):
            return
            
        metrics = self.chess_ai.metrics
        total_games = metrics.total_wins + metrics.total_losses + metrics.total_draws
        win_rate = (metrics.total_wins / total_games * 100) if total_games > 0 else 0
        
        self.metrics_labels["wins"].config(text=str(metrics.total_wins))
        self.metrics_labels["losses"].config(text=str(metrics.total_losses))
        self.metrics_labels["draws"].config(text=str(metrics.total_draws))
        self.metrics_labels["win_rate"].config(text=f"{win_rate:.1f}%")
        self.metrics_labels["games_played"].config(text=str(total_games))

    def update_board(self) -> None:
        """Update the main chess board display."""
        svg_data = chess.svg.board(self.board, size=self.canvas_size)
        png_data = cairosvg.svg2png(bytestring=svg_data.encode())
        image = Image.open(io.BytesIO(png_data))
        self.photo = ImageTk.PhotoImage(image)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        status = f"{turn} to move"
        
        if self.board.is_checkmate():
            status = f"Checkmate! {'Black' if self.board.turn == chess.WHITE else 'White'} wins!"
            if self.has_ai:
                result = 'win' if (self.board.turn != self.player_color) else 'loss'
                self.chess_ai.metrics.update_metrics(result)
                self.update_metrics_display()
        elif self.board.is_stalemate():
            status = "Stalemate!"
            if self.has_ai:
                self.chess_ai.metrics.update_metrics('draw')
                self.update_metrics_display()
        elif self.board.is_check():
            status = f"{turn} is in check!"
            
        self.status_var.set(status)

    def start_self_play_training(self) -> None:
        """Initialize and begin self-play training process."""
        if self.training_in_progress:
            return
            
        try:
            num_games = int(self.num_games_var.get())
            if num_games <= 0:
                raise ValueError("Number of games must be positive")
            
            self.train_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.training_in_progress = True
            self.game_active = False
            self.progress_bar['maximum'] = num_games
            
            training_thread = threading.Thread(target=self.self_play_training, args=(num_games,))
            training_thread.daemon = True
            training_thread.start()
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def on_square_click(self, event: tk.Event) -> None:
        """Handle chess board square clicks."""
        if not self.game_active or self.training_in_progress:
            return
            
        if self.board.turn != self.player_color:
            return
            
        file = event.x // self.square_size
        rank = 7 - (event.y // self.square_size)
        square = chess.square(file, rank)
        
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.highlight_legal_moves(square)
        else:
            move = chess.Move(self.selected_square, square)
            
            # Handle pawn promotion
            if (move in self.board.legal_moves and 
                self.board.piece_at(self.selected_square) and 
                self.board.piece_at(self.selected_square).piece_type == chess.PAWN):
                if rank == 7 or rank == 0:
                    move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
            
            if move in self.board.legal_moves:
                self.board.push(move)
                self.update_board()
                
                if self.has_ai and not self.board.is_game_over():
                    self.root.after(100, self.make_ai_move)
            
            self.selected_square = None
            self.canvas.delete("highlight")

    def highlight_legal_moves(self, square: int) -> None:
        """Highlight squares where the selected piece can move."""
        self.canvas.delete("highlight")
        for move in self.board.legal_moves:
            if move.from_square == square:
                x = (move.to_square & 7) * self.square_size
                y = (7 - (move.to_square >> 3)) * self.square_size
                self.canvas.create_oval(x+self.square_size/3, y+self.square_size/3,
                                          x+self.square_size*2/3, y+self.square_size*2/3,
                                          fill="yellow", tags="highlight")

    def make_ai_move(self) -> None:
        """Make the AI move if it is the AI's turn."""
        if not self.game_active or self.board.is_game_over():
            return
        
        if self.board.turn == self.player_color:
            return
        
        try:
            move = self.chess_ai.get_best_move(self.board)
            if move:
                self.board.push(move)
                self.update_board()
            
            if self.board.is_game_over():
                if self.board.is_checkmate():
                    result = 'win' if self.board.turn != self.player_color else 'loss'
                    self.chess_ai.metrics.update_metrics(result)
                elif self.board.is_stalemate():
                    self.chess_ai.metrics.update_metrics('draw')
        except Exception as e:
            logging.error(f"AI error: {e}")
            messagebox.showerror("Error", f"AI error: {str(e)}")

    def change_color(self) -> None:
        """Handle player color change and restart game."""
        self.new_game()
        self.player_color = chess.WHITE if self.color_var.get() == "white" else chess.BLACK
        if self.player_color == chess.BLACK and self.has_ai:
            self.root.after(500, self.make_ai_move)

    def new_game(self) -> None:
        """Start a new game and reset the board."""
        if self.has_ai and hasattr(self, 'chess_ai') and self.board.move_stack:
            if not self.board.is_game_over():
                self.chess_ai.save_training_state(force=True)

        self.board = chess.Board()
        self.selected_square = None
        self.game_active = True
        self.update_board()
        
        if self.player_color == chess.BLACK:
            self.root.after(500, self.make_ai_move)


def main() -> None:
    """Initialize and run the chess game."""
    root = tk.Tk()
    # Add the custom GUI error logging handler.
    logger = logging.getLogger()
    logger.addHandler(GUIErrorHandler(root))
    app = ChessGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()