"""
Dart Game GUI with automatic and manual input modes.

Features:
- Setup screen: Add players, select game mode
- Game screen: Live scoring with board visualization
- Dual input: Automatic detection OR manual clicking
- Controls: Undo, next player, statistics

Usage:
    python scripts/game_ui.py
    python scripts/game_ui.py -v videos/game.mp4
    python scripts/game_ui.py -c 0 --auto
"""
import cv2
import sys
import argparse
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture import ThreadedCamera, CameraConfig
from src.core import CalibrationData, load_yaml, BoardGeometry, Hit
from src.board import DartboardMapper, BoardVisualizer
from src.detection import HitDetector, HitDetectionConfig
from src.game import GameState, Mode501, ModeTraining, Player
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SetupScreen(ttk.Frame):
    """Setup screen for configuring game."""

    def __init__(self, parent, on_start_callback):
        """
        Initialize setup screen.

        Args:
            parent: Parent widget
            on_start_callback: Callback function(game_state, input_mode) when game starts
        """
        super().__init__(parent)
        self.on_start_callback = on_start_callback

        self.players = []

        self._create_widgets()

    def _create_widgets(self):
        """Create UI widgets."""
        # Title
        title = ttk.Label(
            self,
            text="🎯 Dart Game Setup",
            font=("Arial", 24, "bold")
        )
        title.pack(pady=20)

        # Game Mode Selection
        mode_frame = ttk.LabelFrame(self, text="Game Mode", padding=10)
        mode_frame.pack(fill="x", padx=20, pady=10)

        self.mode_var = tk.StringVar(value="501")

        ttk.Radiobutton(
            mode_frame,
            text="501 (Double Out)",
            variable=self.mode_var,
            value="501"
        ).pack(anchor="w")

        ttk.Radiobutton(
            mode_frame,
            text="Training Mode",
            variable=self.mode_var,
            value="training"
        ).pack(anchor="w")

        # Input Mode Selection
        input_frame = ttk.LabelFrame(self, text="Input Mode", padding=10)
        input_frame.pack(fill="x", padx=20, pady=10)

        self.input_var = tk.StringVar(value="manual")

        ttk.Radiobutton(
            input_frame,
            text="Manual (Click on board)",
            variable=self.input_var,
            value="manual"
        ).pack(anchor="w")

        ttk.Radiobutton(
            input_frame,
            text="Automatic (Detection)",
            variable=self.input_var,
            value="auto"
        ).pack(anchor="w")

        # Player Management
        player_frame = ttk.LabelFrame(self, text="Players", padding=10)
        player_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Player list
        list_frame = ttk.Frame(player_frame)
        list_frame.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.player_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Arial", 12),
            height=6
        )
        self.player_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.player_listbox.yview)

        # Add player controls
        add_frame = ttk.Frame(player_frame)
        add_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(add_frame, text="Name:").pack(side="left", padx=(0, 5))

        self.name_entry = ttk.Entry(add_frame)
        self.name_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.name_entry.bind("<Return>", lambda e: self._add_player())

        ttk.Button(
            add_frame,
            text="Add Player",
            command=self._add_player
        ).pack(side="left", padx=(0, 5))

        ttk.Button(
            add_frame,
            text="Remove",
            command=self._remove_player
        ).pack(side="left")

        # Start button
        start_frame = ttk.Frame(self)
        start_frame.pack(fill="x", padx=20, pady=20)

        self.start_button = ttk.Button(
            start_frame,
            text="Start Game",
            command=self._start_game,
            style="Accent.TButton"
        )
        self.start_button.pack(side="right")

        # Add some default players
        for name in ["Player 1", "Player 2"]:
            self._add_player_internal(name)

    def _add_player(self):
        """Add player from entry."""
        name = self.name_entry.get().strip()

        if not name:
            messagebox.showwarning("Invalid Name", "Please enter a player name")
            return

        if name in self.players:
            messagebox.showwarning("Duplicate Name", f"Player '{name}' already exists")
            return

        self._add_player_internal(name)
        self.name_entry.delete(0, tk.END)

    def _add_player_internal(self, name: str):
        """Add player to list."""
        self.players.append(name)
        self.player_listbox.insert(tk.END, name)

    def _remove_player(self):
        """Remove selected player."""
        selection = self.player_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        name = self.players[idx]

        self.players.pop(idx)
        self.player_listbox.delete(idx)

    def _start_game(self):
        """Start game with current configuration."""
        if not self.players:
            messagebox.showwarning("No Players", "Please add at least one player")
            return

        # Create game state
        game_state = GameState()

        # Set game mode
        mode = self.mode_var.get()
        if mode == "501":
            game_state.game_mode = Mode501(starting_score=501, double_out=True)
        elif mode == "training":
            game_state.game_mode = ModeTraining()

        # Add players
        for name in self.players:
            game_state.add_player(name)

        # Start game
        game_state.start_game()

        # Get input mode
        input_mode = self.input_var.get()

        # Callback
        self.on_start_callback(game_state, input_mode)


class GameScreen(ttk.Frame):
    """Game screen with live scoring."""

    def __init__(
            self,
            parent,
            game_state: GameState,
            input_mode: str,
            camera: ThreadedCamera,
            calibration: CalibrationData,
            mapper: DartboardMapper,
            visualizer: BoardVisualizer,
            hit_detector: HitDetector = None
    ):
        """
        Initialize game screen.

        Args:
            parent: Parent widget
            game_state: Game state object
            input_mode: "manual" or "auto"
            camera: Camera instance
            calibration: Calibration data
            mapper: Board mapper
            visualizer: Board visualizer
            hit_detector: Hit detector (for auto mode)
        """
        super().__init__(parent)

        self.game_state = game_state
        self.input_mode = input_mode
        self.camera = camera
        self.calib = calibration
        self.mapper = mapper
        self.visualizer = visualizer
        self.hit_detector = hit_detector

        # State
        self.running = False
        self.current_frame = None
        self.status_message = ""

        self._create_widgets()
        self._start_update_loop()

    def _create_widgets(self):
        """Create UI widgets."""
        # Main container
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True)

        # Left side: Board view
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Board canvas
        self.board_canvas = tk.Canvas(
            left_frame,
            width=800,
            height=800,
            bg="black"
        )
        self.board_canvas.pack()

        # Bind click for manual mode
        if self.input_mode == "manual":
            self.board_canvas.bind("<Button-1>", self._on_board_click)

        # Status label
        self.status_label = ttk.Label(
            left_frame,
            text="",
            font=("Arial", 12),
            foreground="blue"
        )
        self.status_label.pack(pady=5)

        # Right side: Scores and controls
        right_frame = ttk.Frame(main_container, width=300)
        right_frame.pack(side="right", fill="both", padx=10, pady=10)
        right_frame.pack_propagate(False)

        # Game info
        info_frame = ttk.LabelFrame(right_frame, text="Game Info", padding=10)
        info_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(
            info_frame,
            text=f"Mode: {self.game_state.game_mode.get_name()}",
            font=("Arial", 10)
        ).pack(anchor="w")

        ttk.Label(
            info_frame,
            text=f"Input: {self.input_mode.upper()}",
            font=("Arial", 10)
        ).pack(anchor="w")

        # Current player
        player_frame = ttk.LabelFrame(right_frame, text="Current Player", padding=10)
        player_frame.pack(fill="x", pady=(0, 10))

        self.current_player_label = ttk.Label(
            player_frame,
            text="",
            font=("Arial", 16, "bold")
        )
        self.current_player_label.pack()

        self.current_score_label = ttk.Label(
            player_frame,
            text="",
            font=("Arial", 24, "bold"),
            foreground="green"
        )
        self.current_score_label.pack()

        self.turn_score_label = ttk.Label(
            player_frame,
            text="",
            font=("Arial", 12)
        )
        self.turn_score_label.pack()

        # Scoreboard
        score_frame = ttk.LabelFrame(right_frame, text="Scoreboard", padding=10)
        score_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Scrollable scoreboard
        score_canvas = tk.Canvas(score_frame, height=200)
        scrollbar = ttk.Scrollbar(score_frame, orient="vertical", command=score_canvas.yview)

        self.score_container = ttk.Frame(score_canvas)
        self.score_container.bind(
            "<Configure>",
            lambda e: score_canvas.configure(scrollregion=score_canvas.bbox("all"))
        )

        score_canvas.create_window((0, 0), window=self.score_container, anchor="nw")
        score_canvas.configure(yscrollcommand=scrollbar.set)

        score_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Controls
        control_frame = ttk.LabelFrame(right_frame, text="Controls", padding=10)
        control_frame.pack(fill="x")

        btn_frame1 = ttk.Frame(control_frame)
        btn_frame1.pack(fill="x", pady=(0, 5))

        ttk.Button(
            btn_frame1,
            text="Undo",
            command=self._undo_hit
        ).pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(
            btn_frame1,
            text="Next Player",
            command=self._next_player
        ).pack(side="left", fill="x", expand=True)

        btn_frame2 = ttk.Frame(control_frame)
        btn_frame2.pack(fill="x", pady=(0, 5))

        ttk.Button(
            btn_frame2,
            text="Statistics",
            command=self._show_statistics
        ).pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(
            btn_frame2,
            text="New Game",
            command=self._new_game
        ).pack(side="left", fill="x", expand=True)

    def _start_update_loop(self):
        """Start UI update loop."""
        self.running = True
        self._update()

    def _update(self):
        """Update UI (called periodically)."""
        if not self.running:
            return

        # Update board view
        self._update_board()

        # Update scores
        self._update_scores()

        # Auto mode: Check for hits
        if self.input_mode == "auto" and self.hit_detector and self.current_frame:
            hit = self.hit_detector.detect(self.current_frame)
            if hit:
                self._process_hit(hit)

        # Schedule next update
        self.after(33, self._update)  # ~30 FPS

    def _update_board(self):
        """Update board visualization."""
        # Get frame
        frame = self.camera.read(timeout=0.1)
        if frame is None:
            return

        # Warp to calibrated view
        warped = cv2.warpPerspective(
            frame.image,
            self.calib.homography_matrix,
            (800, 800)
        )

        # Store for detection
        self.current_frame = type('Frame', (), {
            'image': warped,
            'frame_id': frame.frame_id,
            'timestamp': frame.timestamp,
            'fps': frame.fps
        })()

        # Draw overlay
        display = self.visualizer.draw_board_overlay(warped)

        # Draw current turn hits
        player = self.game_state.get_current_player()
        if player:
            hits = player.get_current_turn()
            if hits:
                display = self.visualizer.draw_hits(display, hits, show_scores=True)

        # Convert for Tkinter
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        self.board_canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.board_canvas.image = img_tk  # Keep reference

    def _update_scores(self):
        """Update score display."""
        player = self.game_state.get_current_player()

        if not player:
            return

        # Current player info
        self.current_player_label.config(text=player.name)

        # Score display depends on game mode
        if self.game_state.game_mode.get_starting_score():
            # Countdown mode (501)
            self.current_score_label.config(text=str(player.current_score))
        else:
            # Training mode
            self.current_score_label.config(text=f"Total: {player.total_score}")

        # Current turn
        turn_score = player.get_current_turn_score()
        turn_hits = len(player.get_current_turn())
        self.turn_score_label.config(text=f"This turn: {turn_score} ({turn_hits}/3 darts)")

        # Update scoreboard
        self._update_scoreboard()

        # Status message
        if self.status_message:
            self.status_label.config(text=self.status_message)
            # Clear after 3 seconds
            self.after(3000, lambda: self.status_label.config(text=""))
            self.status_message = ""

    def _update_scoreboard(self):
        """Update full scoreboard."""
        # Clear existing
        for widget in self.score_container.winfo_children():
            widget.destroy()

        # Add all players
        for i, player in enumerate(self.game_state.players):
            is_current = (i == self.game_state.current_player_idx)

            frame = ttk.Frame(
                self.score_container,
                relief="solid" if is_current else "flat",
                borderwidth=2 if is_current else 0
            )
            frame.pack(fill="x", pady=2)

            # Name
            name_label = ttk.Label(
                frame,
                text=player.name,
                font=("Arial", 12, "bold" if is_current else "normal")
            )
            name_label.pack(side="left", padx=5)

            # Score
            if self.game_state.game_mode.get_starting_score():
                score_text = str(player.current_score)
            else:
                score_text = f"{player.total_score} pts"

            score_label = ttk.Label(
                frame,
                text=score_text,
                font=("Arial", 12)
            )
            score_label.pack(side="right", padx=5)

    def _on_board_click(self, event):
        """Handle board click (manual mode)."""
        x, y = event.x, event.y

        # Calculate hit
        hit = self.mapper.pixel_to_score(float(x), float(y))

        self._process_hit(hit)

    def _process_hit(self, hit: Hit):
        """Process a hit (from auto or manual)."""
        if self.game_state.game_finished:
            return

        player = self.game_state.get_current_player()
        if not player:
            return

        # Add hit
        continue_turn = self.game_state.add_hit(hit)

        logger.info(f"{player.name}: {hit.score} points")

        # Check if turn is complete
        if not continue_turn:
            self._finalize_turn()

    def _finalize_turn(self):
        """Finalize current turn."""
        message = self.game_state.finalize_turn()

        if message:
            self.status_message = message

            # Check for winner
            if self.game_state.game_finished:
                self._show_winner()

    def _next_player(self):
        """Manually advance to next player."""
        player = self.game_state.get_current_player()
        if not player:
            return

        # Check if turn has hits
        if player.get_current_turn():
            # Finalize turn
            self._finalize_turn()
        else:
            # Just advance
            self.game_state.next_player()

    def _undo_hit(self):
        """Undo last hit."""
        if self.game_state.undo_last_hit():
            self.status_message = "Hit undone"
        else:
            self.status_message = "Nothing to undo"

    def _show_statistics(self):
        """Show player statistics."""
        stats_window = tk.Toplevel(self)
        stats_window.title("Statistics")
        stats_window.geometry("400x300")

        # Create stats display
        for player in self.game_state.players:
            frame = ttk.LabelFrame(stats_window, text=player.name, padding=10)
            frame.pack(fill="x", padx=10, pady=5)

            stats = [
                f"Current Score: {player.current_score}",
                f"Darts Thrown: {player.darts_thrown}",
                f"Total Scored: {player.total_score}",
                f"Average/Dart: {player.average_per_dart:.1f}",
                f"Average/Turn: {player.average_per_turn:.1f}",
                f"Highest Turn: {player.highest_turn}",
            ]

            for stat in stats:
                ttk.Label(frame, text=stat).pack(anchor="w")

    def _show_winner(self):
        """Show winner dialog."""
        winner = self.game_state.winner
        if not winner:
            return

        messagebox.showinfo(
            "Game Over!",
            f"🎉 {winner.name} wins!\n\n"
            f"Darts thrown: {winner.darts_thrown}\n"
            f"Average: {winner.average_per_dart:.1f} per dart"
        )

    def _new_game(self):
        """Start a new game."""
        response = messagebox.askyesno(
            "New Game",
            "Start a new game? Current game will be lost."
        )

        if response:
            self.running = False
            self.master.show_setup()

    def stop(self):
        """Stop update loop."""
        self.running = False


class DartGameApp(tk.Tk):
    """Main application window."""

    def __init__(
            self,
            camera: ThreadedCamera,
            calibration: CalibrationData
    ):
        """
        Initialize application.

        Args:
            camera: Camera instance
            calibration: Calibration data
        """
        super().__init__()

        self.title("🎯 Dart Game")
        self.geometry("1200x900")

        self.camera = camera
        self.calibration = calibration

        # Initialize mapper and visualizer
        self.mapper = DartboardMapper(
            board_geometry=BoardGeometry(),
            board_center=calibration.board_center,
            mm_per_pixel=calibration.mm_per_pixel
        )

        self.visualizer = BoardVisualizer(self.mapper, opacity=0.3)

        # Screens
        self.setup_screen = None
        self.game_screen = None

        # Show setup
        self.show_setup()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def show_setup(self):
        """Show setup screen."""
        if self.game_screen:
            self.game_screen.stop()
            self.game_screen.destroy()
            self.game_screen = None

        self.setup_screen = SetupScreen(self, self._on_start_game)
        self.setup_screen.pack(fill="both", expand=True)

    def _on_start_game(self, game_state: GameState, input_mode: str):
        """Start game callback."""
        # Remove setup screen
        if self.setup_screen:
            self.setup_screen.destroy()
            self.setup_screen = None

        # Create hit detector for auto mode
        hit_detector = None
        if input_mode == "auto":
            hit_detector = HitDetector(self.mapper, config=HitDetectionConfig())

        # Show game screen
        self.game_screen = GameScreen(
            self,
            game_state=game_state,
            input_mode=input_mode,
            camera=self.camera,
            calibration=self.calibration,
            mapper=self.mapper,
            visualizer=self.visualizer,
            hit_detector=hit_detector
        )
        self.game_screen.pack(fill="both", expand=True)

    def _on_closing(self):
        """Handle window close."""
        if self.game_screen:
            self.game_screen.stop()

        self.camera.stop()
        self.destroy()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dart game with automatic or manual scoring"
    )

    parser.add_argument(
        "--calib",
        type=str,
        default="config/calib.yaml",
        help="Calibration file"
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "-c", "--camera",
        type=int,
        default=None,
        help="Camera index"
    )
    source_group.add_argument(
        "-v", "--video",
        type=str,
        default=None,
        help="Video file path"
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help="Start in automatic mode"
    )

    return parser.parse_args()


def main():
    """Run dart game application."""
    args = parse_args()

    # Load calibration
    calib_path = Path(args.calib)
    if not calib_path.exists():
        logger.error(f"Calibration file not found: {calib_path}")
        logger.info("Run 'python scripts/calibrate.py' first")
        return

    try:
        calib_dict = load_yaml(calib_path)
        calib_data = CalibrationData(
            homography_matrix=np.array(calib_dict["calibration"]["homography_matrix"]),
            board_center=tuple(calib_dict["calibration"]["board_center"]),
            mm_per_pixel=calib_dict["calibration"]["mm_per_pixel"],
            board_radius_px=calib_dict["calibration"]["board_radius_px"],
            method=calib_dict["calibration"]["method"],
            timestamp=calib_dict["calibration"].get("timestamp")
        )
        logger.info(f"Calibration loaded: {calib_data.method}")
    except Exception as e:
        logger.error(f"Failed to load calibration: {e}")
        return

    # Determine source
    if args.video:
        source = args.video
        loop = True
    elif args.camera is not None:
        source = args.camera
        loop = False
    else:
        source = 0
        loop = False

    # Initialize camera
    cam_config = CameraConfig(source=source, loop_video=loop)
    camera = ThreadedCamera(config=cam_config, queue_size=3)

    if not camera.start():
        logger.error("Failed to start camera/video")
        return

    try:
        # Run application
        app = DartGameApp(camera, calib_data)
        app.mainloop()

    finally:
        camera.stop()


if __name__ == "__main__":
    main()