use chrono::{DateTime, Utc};

// Maximum number of turns to keep in history by default
const DEFAULT_MAX_HISTORY_TURNS: usize = 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Speaker {
    User,
    Model,
    System, // For system messages, e.g., connection status, errors
}

#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub speaker: Speaker,
    pub text: String,
    pub timestamp: DateTime<Utc>,
    // Optional fields for future use:
    // pub audio_tokens: Option<Vec<Vec<u32>>>,
    // pub duration_ms: Option<u64>,
    // pub emotion: Option<String>,
    // pub style: Option<String>,
}

impl ConversationTurn {
    pub fn new(speaker: Speaker, text: String) -> Self {
        Self {
            speaker,
            text,
            timestamp: Utc::now(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConversationHistory {
    turns: Vec<ConversationTurn>,
    max_turns: usize,
}

impl ConversationHistory {
    pub fn new(max_turns: Option<usize>) -> Self {
        Self {
            turns: Vec::new(),
            max_turns: max_turns.unwrap_or(DEFAULT_MAX_HISTORY_TURNS),
        }
    }

    pub fn add_turn(&mut self, turn: ConversationTurn) {
        self.turns.push(turn);
        self.enforce_limit();
    }

    fn enforce_limit(&mut self) {
        let current_len = self.turns.len();
        if current_len > self.max_turns {
            let excess = current_len - self.max_turns;
            // Remove the oldest turns
            self.turns.drain(0..excess);
        }
    }

    pub fn get_turns(&self) -> &[ConversationTurn] {
        &self.turns
    }

    pub fn clear(&mut self) {
        self.turns.clear();
    }

    /// Formats the conversation history into a string suitable for a model prompt,
    /// respecting a maximum character limit.
    /// Iterates backwards from the most recent turn.
    pub fn format_for_prompt(&self, max_chars: usize) -> String {
        let mut prompt = String::new();
        let mut current_chars = 0;

        for turn in self.turns.iter().rev() {
            let prefix = match turn.speaker {
                Speaker::User => "User:",
                Speaker::Model => "Model:",
                Speaker::System => "System:", // Decide if System turns should be included
            };
            // Simple formatting: Prefix<space>Text<newline>
            let formatted_turn = format!("{} {}\n", prefix, turn.text);
            let turn_chars = formatted_turn.chars().count();

            // Check if adding this turn exceeds the limit
            if current_chars + turn_chars > max_chars {
                // If even the first turn is too long, we can't include anything.
                // Or, maybe we should try to truncate the turn itself? For now, just stop.
                break;
            }

            // Prepend the turn to maintain chronological order in the final string
            prompt.insert_str(0, &formatted_turn);
            current_chars += turn_chars;
        }

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_add_turn() {
        let mut history = ConversationHistory::new(Some(3));
        history.add_turn(ConversationTurn::new(Speaker::User, "Hello".to_string()));
        assert_eq!(history.get_turns().len(), 1);
        history.add_turn(ConversationTurn::new(Speaker::Model, "Hi there".to_string()));
        assert_eq!(history.get_turns().len(), 2);
    }

    #[test]
    fn test_history_limit() {
        let mut history = ConversationHistory::new(Some(2));
        let turn1 = ConversationTurn::new(Speaker::User, "First".to_string());
        let turn2 = ConversationTurn::new(Speaker::Model, "Second".to_string());
        let turn3 = ConversationTurn::new(Speaker::User, "Third".to_string());

        history.add_turn(turn1.clone());
        history.add_turn(turn2.clone());
        history.add_turn(turn3.clone());

        assert_eq!(history.get_turns().len(), 2);
        // Check that the oldest turn (turn1) was removed
        assert_eq!(history.get_turns()[0].text, "Second");
        assert_eq!(history.get_turns()[1].text, "Third");
    }

    #[test]
    fn test_clear_history() {
        let mut history = ConversationHistory::new(Some(5));
        history.add_turn(ConversationTurn::new(Speaker::User, "Test 1".to_string()));
        history.add_turn(ConversationTurn::new(Speaker::Model, "Test 2".to_string()));
        assert!(!history.get_turns().is_empty());
        history.clear();
        assert!(history.get_turns().is_empty());
    }

    #[test]
    fn test_format_prompt() {
        let mut history = ConversationHistory::new(Some(5)); // Max turns doesn't matter here
        history.add_turn(ConversationTurn::new(Speaker::User, "Hello".to_string()));
        sleep(Duration::from_millis(1)); // Ensure timestamps differ
        history.add_turn(ConversationTurn::new(Speaker::Model, "Hi there.".to_string()));
        sleep(Duration::from_millis(1));
        history.add_turn(ConversationTurn::new(Speaker::User, "How are you?".to_string()));

        // Expected format (most recent first in iteration, prepended)
        // User: Hello
        // Model: Hi there.
        // User: How are you?
        let expected_full = "User: Hello\nModel: Hi there.\nUser: How are you?\n";
        
        // Test with limit large enough for all turns
        assert_eq!(history.format_for_prompt(1000), expected_full);

        // Test with limit allowing only the last two turns
        // "Model: Hi there.\n" (16 chars) + "User: How are you?\n" (19 chars) = 35 chars
        let expected_partial = "Model: Hi there.\nUser: How are you?\n";
        assert_eq!(history.format_for_prompt(40), expected_partial);
        assert_eq!(history.format_for_prompt(35), expected_partial);

        // Test with limit allowing only the last turn
        // "User: How are you?\n" (19 chars)
        let expected_last = "User: How are you?\n";
        assert_eq!(history.format_for_prompt(20), expected_last);
        assert_eq!(history.format_for_prompt(19), expected_last);
        
        // Test with limit too small for any turn
        assert_eq!(history.format_for_prompt(10), "");
        assert_eq!(history.format_for_prompt(0), "");
    }
} 