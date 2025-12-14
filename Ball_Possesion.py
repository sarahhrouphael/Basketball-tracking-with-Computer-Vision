import numpy as np

SCORE_COOLDOWN = 15


class BallPossessionTracker:
    def __init__(self, overlap_thresh=0.6, cooldown_frames=5):
        self.overlap_thresh = overlap_thresh
        self.cooldown_frames = cooldown_frames

        self.current_team = None
        self.last_change_frame = -999

        # MISSING (needed for scoring cooldown)
        self.last_score_frame = -999


    @staticmethod
    def _intersection_area(a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])

        if xA >= xB or yA >= yB:
            return 0.0

        return (xB - xA) * (yB - yA)

    def _ball_coverage(self, ball_box, player_box):
        inter = self._intersection_area(ball_box, player_box)
        ball_area = (ball_box[2] - ball_box[0]) * (ball_box[3] - ball_box[1])
        if ball_area <= 0:
            return 0.0
        return inter / ball_area

    def update(self, ball_box, player_boxes, track_team, frame_idx):
        """
        ball_box: (x1,y1,x2,y2) or None
        player_boxes: dict {track_id: (x1,y1,x2,y2)}
        track_team: dict {track_id: team_id}
        """

        if ball_box is None:
            self.current_team = None
            return None

        best_tid = None
        best_score = 0.0

        for tid, pbox in player_boxes.items():
            score = self._ball_coverage(ball_box, pbox)
            if score > best_score:
                best_score = score
                best_tid = tid

        if best_score < self.overlap_thresh:
            self.current_team = None
            return None

        team = track_team.get(best_tid, None)

        # cooldown to avoid flickering
        if (
            team is not None and
            team != self.current_team and
            frame_idx - self.last_change_frame >= self.cooldown_frames
        ):
            self.current_team = team
            self.last_possession_team = team  
            self.last_change_frame = frame_idx

        return self.current_team

