"""Gradio UI for the AI Travel Concierge built on the agent framework.

This module wires together the `TravelAgent` with a custom-styled Gradio app.
It provides:
- A chat interface with streaming assistant responses
- An events side panel showing tool calls/results as compact cards
- User profile switching (based on `context/seed.json`)
- Calendar (.ics) open/folder helpers for macOS/Windows/Linux
"""
import warnings
warnings.filterwarnings("ignore")
import asyncio
import gradio as gr
import os
import subprocess
import platform
import json
from typing import List, Dict
from pathlib import Path

from agent import TravelAgent
from config import get_config, validate_dependencies


def load_css() -> str:
    """Load CSS from external file."""
    css_path = Path(__file__).parent / "assets" / "styles.css"
    try:
        return css_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Warning: CSS file not found at {css_path}")
        return ""


def open_calendar_file(file_path: str) -> tuple[str, bool]:
    """Open a calendar file with the system's default calendar application.
    
    Args:
        file_path: Path to the .ics calendar file
        
    Returns:
        Tuple of (message, success) where message describes the result
    """
    if not file_path or not os.path.exists(file_path):
        return "❌ Calendar file not found", False
    
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            # Use 'open' command to open with default calendar app (Calendar.app)
            subprocess.run(["open", file_path], check=True)
            return "📅 Calendar opened with Calendar.app", True
        elif system == "Windows":
            # Use 'start' command to open with default calendar app
            subprocess.run(["start", file_path], shell=True, check=True)
            return "📅 Calendar opened with default calendar app", True
        elif system == "Linux":
            # Use 'xdg-open' to open with default calendar app
            subprocess.run(["xdg-open", file_path], check=True)
            return "📅 Calendar opened with default calendar app", True
        else:
            return f"❓ Unsupported system: {system}", False
    except subprocess.CalledProcessError as e:
        return f"❌ Failed to open calendar: {e}", False
    except FileNotFoundError:
        if system == "Linux":
            return "❌ xdg-open not found. Please install xdg-utils", False
        else:
            return "❌ System command not found", False
    except Exception as e:
        return f"❌ Error opening calendar: {e}", False


def open_calendar_folder(folder_path: str) -> tuple[str, bool]:
    """Open the calendars folder with the system's default file explorer.
    
    Args:
        folder_path: Path to the calendars folder
        
    Returns:
        Tuple of (message, success) where message describes the result
    """
    try:
        if not folder_path or not os.path.isdir(folder_path):
            return "⚠️ No calendar folder found yet. Generate a calendar to create it.", False
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["open", folder_path], check=True)
            return "📂 Opened calendars folder in Finder", True
        elif system == "Windows":
            subprocess.run(["explorer", folder_path], check=True)
            return "📂 Opened calendars folder in Explorer", True
        elif system == "Linux":
            subprocess.run(["xdg-open", folder_path], check=True)
            return "📂 Opened calendars folder", True
        else:
            return f"❓ Unsupported system: {system}", False
    except subprocess.CalledProcessError as e:
        return f"❌ Failed to open folder: {e}", False
    except FileNotFoundError:
        return "❌ System command not found", False
    except Exception as e:
        return f"❌ Error opening folder: {e}", False


class TravelAgentUI:
    """
    Gradio UI wrapper for the TravelAgent.
    """
    
    def __init__(self, config=None):
        """Initialize the UI with a travel agent instance."""
        self.config = config or get_config()
        self.agent = TravelAgent(config=self.config)
        self.current_user_id = self._get_default_user_id_from_seed()
        # Number of past conversation steps (user+assistant pairs) to render
        self.history_steps = 20
        self.initial_history = []  # Will be populated async
        self.user_ids = []  # Dynamic users loaded from seed
        self.latest_calendar_file = None  # Track latest generated calendar
    
    def _get_default_user_id_from_seed(self) -> str:
        """Derive default user ID from first user in context/seed.json.
        Falls back to "Tyler" if unavailable.
        """
        try:
            seed_path = Path(__file__).parent / "context" / "seed.json"
            data = json.loads(seed_path.read_text(encoding="utf-8"))
            user_memories = data.get("user_memories", {})
            if isinstance(user_memories, dict) and user_memories:
                # Python 3.7+ preserves insertion order of dict keys
                return next(iter(user_memories.keys()))
        except Exception as e:
            print(f"⚠️ Warning: Could not determine default user from seed: {e}")
        return "Tyler"
    
    async def initialize_chat_history(self):
        """Initialize seed data and load initial chat history for the default user."""
        try:
            # Init seed data for all users
            await self.agent.initialize_seed_data()
            users = self.agent.get_all_user_ids()
            # Sort users but ensure current user is first if present
            sorted_users = sorted(users)
            if self.current_user_id in sorted_users:
                sorted_users.remove(self.current_user_id)
                self.user_ids = [self.current_user_id] + sorted_users
            else:
                self.user_ids = sorted_users
            # Load last N steps (approx 2 messages per step: user + assistant)
            self.initial_history = await self.agent.get_chat_history(
                self.current_user_id, n=self.history_steps * 2
            )
            print(f"✅ Loaded {len(self.initial_history)} initial messages for user: {self.current_user_id}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load initial chat history: {e}")
            self.initial_history = []

    async def switch_user(self, new_user_id: str) -> List[dict]:
        """Switch to a different user profile and load their chat history.
        
        Args:
            new_user_id: The user ID to switch to
            
        Returns:
            Chat history for the new user
        """
        if new_user_id == self.current_user_id:
            # No change needed, return current history (last N steps)
            return await self.agent.get_chat_history(self.current_user_id, n=self.history_steps * 2)
        
        # Switch to the new user
        self.current_user_id = new_user_id
        
        # Initialize the new user context (this will preseed if needed)
        self.agent._get_or_create_user_ctx(new_user_id)
        
        # Return the new user's last N steps of chat history
        return await self.agent.get_chat_history(new_user_id, n=self.history_steps * 2)
    
    async def clear_chat_history(self) -> List[dict]:
        """Clear chat history for the current user from Redis and UI."""
        if not self.current_user_id:
            return []
        try:
            # Get the user context and clear its Redis history
            ctx = self.agent._get_or_create_user_ctx(self.current_user_id)
            await ctx.agent.model_context.clear()
            print(f"✅ Cleared chat history for user: {self.current_user_id}")
            return []
        except Exception as e:
            print(f"❌ Error clearing chat history for user {self.current_user_id}: {e}")
            return []
    
    def create_interface(self) -> gr.Interface:
        """Create and return the Gradio interface."""
        
        # Load CSS from external file
        css = load_css()
        
        # Create custom Redis theme
        redis_theme = gr.themes.Soft(
            font=[
                gr.themes.GoogleFont("Space Grotesk"),
                "ui-sans-serif",
                "system-ui", 
                "sans-serif"
            ],
            font_mono=[
                gr.themes.GoogleFont("Space Mono"),
                "ui-monospace",
                "Consolas",
                "monospace"
            ],
            primary_hue=gr.themes.colors.yellow,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate
        ).set(
            # Background colors
            background_fill_primary="#091A23",
            background_fill_secondary="#163341",
            background_fill_primary_dark="#091A23",
            background_fill_secondary_dark="#163341",
            
            # Button colors
            button_primary_background_fill="#DCFF1E",
            button_primary_background_fill_dark="#DCFF1E",
            button_primary_text_color="#091A23",
            button_primary_text_color_dark="#091A23",
            button_primary_background_fill_hover="#091A23",
            button_primary_background_fill_hover_dark="#091A23",
            button_primary_text_color_hover="#FFFFFF",
            button_primary_text_color_hover_dark="#FFFFFF",
            button_primary_border_color_hover="#091A23",
            
            button_secondary_background_fill="#5C707A",
            button_secondary_background_fill_hover="#2D4754",
            button_secondary_border_color="#5C707A",
            button_secondary_border_color_hover="#2D4754",
            button_secondary_text_color="#FFFFFF",
            button_secondary_text_color_hover="#FFFFFF",
            
            # Input and text colors
            input_background_fill="#091A23",
            body_text_color="#FFFFFF",
            checkbox_label_background_fill="#091A23",
            checkbox_label_text_color="#FFFFFF",
            checkbox_label_text_color_selected="#FFFFFF",
            
            # Accent colors
            color_accent_soft="#163341",
            border_color_accent_subdued="#FFFFFF",
            
            # Panel colors
            panel_border_color_dark="#8A99A0",
            panel_border_width_dark="1px",
            panel_background_fill_dark="#091A23",
            
            # Stat colors
            stat_background_fill="#C795E3",
            stat_background_fill_dark="#C795E3"
        )
        
        # Create the chat interface
        with gr.Blocks(
            title="AI Travel Concierge",
            theme=redis_theme,
            css=css
        ) as interface:
            
            # Header
            gr.HTML("""
                <div style="text-align: center; margin: 8px 0; font-family: 'Space Grotesk', sans-serif;">
                    <h1 style="color: #FFFFFF; margin-bottom: 10px; font-weight: 500;">🌍 AI Travel Concierge</h1>
                    <p style="color: #DCFF1E; font-size: 18px; font-weight: 500;">
                        Your intelligent travel planning assistant powered by AI & Long Term Memory
                    </p>
                    <p style="color: #8A99A0; font-size: 14px;">
                        Built on Redis, Mem0, Microsoft Agent Framework, Tavily, and OpenAI
                    </p>
                </div>
            """)
            
            # User Profile Switcher (top)
            with gr.Row(elem_classes=["user-selector-container"], equal_height=True):
                user_tab_components = {}
                # Default selected is the first tab; ensure user_ids prepared in initialize_chat_history
                with gr.Tabs(selected=0) as user_tabs:
                    for uid in (self.user_ids or [self.current_user_id]):
                        with gr.Tab(uid) as _tab:
                            user_tab_components[uid] = _tab
            
            # Two-column layout: Chat (left 70%) and Agent Logs (right 30%)
            with gr.Row(equal_height=False, variant="panel"):
                # Chat Column (70% width)
                with gr.Column(scale=7, min_width=0):
                    
                    gr.HTML("""
                        <div style="text-align: center; margin-bottom: 15px; padding: 15px; background: linear-gradient(135deg, #163341 0%, #091A23 100%); border-radius: 8px;">
                            <h3 style="color: #DCFF1E; margin: 0; font-size: 16px;">💬 Chat</h3>
                        </div>
                    """)
                    
                    chatbot = gr.Chatbot(
                        value=self.initial_history,
                        show_label=False,
                        container=True,
                        type="messages",
                        avatar_images=None,
                        show_copy_button=False,
                        height=500,
                        autoscroll=True
                    )
                    
                    # Input components
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask me about planning your next trip...",
                            show_label=False,
                            container=False,
                            scale=5,
                            lines=1,
                            interactive=True
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1, interactive=True)
                    
                    # Control buttons
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")
                        calendar_open_btn = gr.Button(
                            "📅 Open Calendar", 
                            variant="secondary", 
                            size="sm",
                            visible=True
                        )
                    
                    # Status message for calendar operations
                    calendar_status = gr.Textbox(
                        value="",
                        show_label=False,
                        container=False,
                        interactive=False,
                        visible=False,
                        elem_classes=["calendar-status"]
                    )

                # Agent Logs Panel (30% width)
                with gr.Column(scale=3, min_width=0):
                    gr.HTML("""
                        <div style="text-align: center; margin-bottom: 15px; padding: 15px; background: linear-gradient(135deg, #163341 0%, #091A23 100%); border-radius: 8px;">
                            <h3 style="color: #DCFF1E; margin: 0; font-size: 16px;">🛈 Agent Logs</h3>
                        </div>
                    """)
                    events_state = gr.State([])
                    calendar_file_state = gr.State(None)
                    events_panel = gr.HTML(
                        value="""
                        <div style="height: 500px; padding: 15px; background: #163341; border-radius: 8px; color: #8A99A0; overflow-y: auto;">
                            <div style="text-align: center; margin-top: 200px;">
                                <p>🤖 Agent events will appear here during chat</p>
                            </div>
                        </div>
                        """, 
                        elem_id="events-panel"
                    )
            
            # Event handler functions
            async def handle_streaming_chat(message, history, events, calendar_file):
                """Handle streaming chat and side-panel event rendering."""
                if not message.strip():
                    events_html = ""
                    yield history, events, events_html, calendar_file, gr.update(visible=True), gr.update(value="", visible=False)
                    return
                
                # Add user message to history and show it immediately
                history = history + [{"role": "user", "content": message}]
                # Clear events at the start of each new submit
                events = []
                yield history, events, "", calendar_file, gr.update(visible=True), gr.update(value="", visible=False)
                
                # Initialize assistant message with animated thinking dots
                history = history + [{"role": "assistant", "content": '<span class="thinking-animation">●●●</span>'}]
                events_html = "".join([e.get("html", "") for e in events[-200:]])
                yield history, events, events_html, calendar_file, gr.update(visible=True), gr.update(value="", visible=False)
                
                final_response = ""
                calendar_generated = False
                latest_calendar_file = calendar_file
                
                try:
                    # Stream the response with events
                    async for partial_response, evt in self.agent.stream_chat_turn_with_events(self.current_user_id, message):
                        # Keep the thinking dots visible while streaming
                        history[-1] = {"role": "assistant", "content": partial_response}
                        final_response = partial_response  # Keep track of final response
                        
                        # Check if this is a calendar generation event
                        if evt and evt.get("type") == "tool_result" and evt.get("tool_name") == "generate_calendar_ics":
                            calendar_generated = True
                            if evt.get("file_path"):
                                latest_calendar_file = evt.get("file_path")
                                self.latest_calendar_file = latest_calendar_file
                        
                        # Append new event if any
                        if evt is not None:
                            events = events + [evt]
                        # Render compact HTML for events
                        events_html = "".join([e.get("html", "") for e in events[-200:]])
                        
                        # Update calendar download visibility
                        yield history, events, events_html, latest_calendar_file, gr.update(visible=True), gr.update(value="", visible=False)
                    
                    # After streaming completes, store conversation memory asynchronously
                    if final_response and not final_response.endswith('●●●</span>'):
                        # Create a background task to store memory without blocking
                        import asyncio
                        asyncio.create_task(
                            self.agent.store_memory(
                                self.current_user_id, 
                                message, 
                            )
                        )
                    
                    # Final yield with calendar file info
                    yield history, events, events_html, latest_calendar_file, gr.update(visible=True), gr.update(value="", visible=False)
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    # Replace thinking indicator with error message
                    history[-1] = {"role": "assistant", "content": error_msg}
                    print(f"❌ Error during streaming chat: {e}", flush=True)
                    events_html = "".join([e.get("html", "") for e in events[-200:]])
                    yield history, events, events_html, latest_calendar_file, gr.update(visible=True), gr.update(value="", visible=False)
            
            
            # Event handlers for chat - streaming version
            async def handle_submit_and_clear(message, history, events, calendar_file):
                """Handle message submission and immediately clear input."""
                async for h, ev, html, cal_file, btn_update, status_update in handle_streaming_chat(message, history, events, calendar_file):
                    yield h, ev, html, "", cal_file, btn_update, status_update
            
            msg.submit(
                handle_submit_and_clear,
                [msg, chatbot, events_state, calendar_file_state],
                [chatbot, events_state, events_panel, msg, calendar_file_state, calendar_open_btn, calendar_status],
                queue=True
            )
            
            send_btn.click(
                handle_submit_and_clear,
                [msg, chatbot, events_state, calendar_file_state],
                [chatbot, events_state, events_panel, msg, calendar_file_state, calendar_open_btn, calendar_status],
                queue=True
            )
            
            # User switcher handlers via Tabs (dynamic)
            def make_handle_tab(user_id: str):
                async def _handler():
                    try:
                        history = await self.switch_user(user_id)
                        # Reset events panel to initial state
                        initial_events_html = """
                        <div style="height: 500px; padding: 15px; background: #163341; border-radius: 8px; color: #8A99A0; overflow-y: auto;">
                            <div style="text-align: center; margin-top: 200px;">
                                <p>🤖 Agent events will appear here during chat</p>
                            </div>
                        </div>
                        """
                        return history, [], initial_events_html, "", None, gr.update(visible=True), gr.update(value="", visible=False)
                    except Exception as e:
                        print(f"Error switching to user {user_id}: {e}")
                        return [], [], "", "", None, gr.update(visible=True), gr.update(value="", visible=False)
                return _handler

            # Attach selection handlers for each user tab
            for uid, tab in (user_tab_components or {}).items():
                tab.select(
                    make_handle_tab(uid),
                    outputs=[chatbot, events_state, events_panel, msg, calendar_file_state, calendar_open_btn, calendar_status],
                    queue=True
                )
            
            # Clear chat functionality
            async def handle_clear_chat():
                """Handle clearing chat history from Redis and UI."""
                cleared_history = await self.clear_chat_history()
                # Reset events panel to initial state
                initial_events_html = """
                <div style="height: 500px; padding: 15px; background: #163341; border-radius: 8px; color: #8A99A0; overflow-y: auto;">
                    <div style="text-align: center; margin-top: 200px;">
                        <p>🤖 Agent events will appear here during chat</p>
                    </div>
                </div>
                """
                return cleared_history, [], initial_events_html, "", None, gr.update(visible=True), gr.update(value="", visible=False)
            
            clear_btn.click(
                handle_clear_chat,
                outputs=[chatbot, events_state, events_panel, msg, calendar_file_state, calendar_open_btn, calendar_status],
                queue=True
            )
            
            # Calendar open functionality
            def handle_calendar_open():
                """Open the current user's calendars folder in the system file explorer."""
                folder_path = str((Path(__file__).parent / "assets" / "calendars" / self.current_user_id).absolute())
                message, success = open_calendar_folder(folder_path)
                return gr.update(value=message, visible=True)
            
            calendar_open_btn.click(
                handle_calendar_open,
                inputs=[],
                outputs=[calendar_status],
                queue=False
            )
        
        return interface


async def create_app(config=None) -> gr.Interface:
    """Create and return the Gradio app."""
    if config is None:
        config = get_config()
    ui = TravelAgentUI(config=config)
    # Initialize chat history before creating interface
    await ui.initialize_chat_history()
    return ui.create_interface()


def main():
    """Main function to launch the Gradio app."""
    # Set environment variable to suppress warnings before any imports
    os.environ['PYTHONWARNINGS'] = 'ignore'    
    print("🌍 AI Travel Concierge - Starting up...")
    
    # Load and validate configuration
    config = get_config()
    print("✅ Configuration loaded successfully")
    
    # Validate dependencies
    if not validate_dependencies():
        print("\n❌ Dependency validation failed. Please check your configuration.")
        return
    
    # Create and launch the app
    try:
        # Use asyncio to handle async app creation
        import asyncio
        app = asyncio.run(create_app(config))
        
        print(f"\n🚀 Launching AI Travel Concierge on {config.server_name}:{config.server_port}")
        
        app.queue().launch(
            server_name=config.server_name,
            server_port=config.server_port,
            share=config.share,
            show_error=True,
            inbrowser=True,
            debug=True
        )

    except Exception as e:
        print(f"❌ Failed to launch application: {e}")
        raise


if __name__ == "__main__":
    main()
