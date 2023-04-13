import numpy as np
from matplotlib.lines import Line2D


def normalize_xydata(line: Line2D) -> tuple[np.ndarray, np.ndarray]:
    """Make sure datetime values are properly converted to floats and convert
    into data coordinates."""
    # Convert the data into the data coordinates
    line_t = line.get_transform().transform
    axes_t = line.axes.transData.inverted().transform
    x, y = axes_t(line_t(line.get_xydata())).T

    return x, y


# From https://www.geeksforgeeks.org/maximum-bipartite-matching/
class GFG:
    def __init__(self, graph):
        # residual graph
        self.graph = graph
        self.ppl = len(graph)
        self.jobs = len(graph[0])

    # A DFS based recursive function that returns true if a matching for vertex
    # u is possible
    def bpm(self, u, match_r, seen):
        # Try every job one by one
        for v in range(self.jobs):
            # If applicant u is interested
            # in job v and v is not seen
            if self.graph[u][v] and not seen[v]:
                # Mark v as visited
                seen[v] = True

                # If job 'v' is not assigned to an applicant OR previously
                # assigned applicant for job v (which is matchR[v]) has an
                # alternate job available. Since v is marked as visited in the
                # above line, matchR[v] in the following recursive call will not
                # get job 'v' again

                if match_r[v] == -1 or self.bpm(match_r[v], match_r, seen):
                    match_r[v] = u
                    return True
        return False

    # Returns maximum number of matching
    def maxBPM(self):
        # An array to keep track of the applicants assigned to jobs. The value
        # of matchR[i] is the applicant number assigned to job i, the value -1
        # indicates nobody is assigned.
        matchR = [-1] * self.jobs

        # Count of jobs assigned to applicants
        result = 0
        for i in range(self.ppl):
            # Mark all jobs as not seen for next applicant.
            seen = [False] * self.jobs

            # Find if the applicant 'u' can get a job
            if self.bpm(i, matchR, seen):
                result += 1
        return result, matchR


def maximum_bipartite_matching(graph: np.ndarray) -> np.ndarray:
    """Finds the maximum bipartite matching of a graph

    Parameters
    ----------
    graph : np.ndarray
        The graph, represented as a boolean matrix

    Returns
    -------
    order : np.ndarray
        The order in which to traverse the graph to visit a maximum of nodes
    """
    g = GFG(graph)
    _, order = g.maxBPM()
    return np.asarray(order)
