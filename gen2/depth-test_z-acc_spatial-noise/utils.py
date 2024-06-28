import numpy as np

def normalize(v):
	norm = np.linalg.norm(v)
	if norm == 0:
		return v
	return v / norm


def angle(v1, v2, ax):
	u1 = v1 - v1 @ ax * ax
	u2 = v2 - v2 @ ax * ax

	a = np.arccos(np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2)))
	dir = -1 if np.cross(u1, u2) @ ax < 0 else 1
	return a * dir