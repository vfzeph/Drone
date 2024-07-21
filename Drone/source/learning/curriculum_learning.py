class CurriculumLearning:
    def __init__(self, initial_difficulty, difficulty_increment, difficulty_threshold):
        self.current_difficulty = initial_difficulty
        self.difficulty_increment = difficulty_increment
        self.difficulty_threshold = difficulty_threshold

    def adjust_difficulty(self, performance_metric):
        if performance_metric > self.difficulty_threshold:
            self.current_difficulty += self.difficulty_increment
        return self.current_difficulty

    def get_current_difficulty(self):
        return self.current_difficulty
