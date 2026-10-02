export type Metric = 'euclidean' | 'cosine' | 'dot';

export interface SearchOptions {
  /** candidate list size during layer-0 search; higher = better recall, slower */
  ef?: number;
}

export interface SearchResult {
  id: string;
  dist: number;
}

export interface SearchHop {
  from: string;
  to: string;
  dist: number;
}

export interface TraceLayer {
  level: number;
  entry: string | null;
  hops: SearchHop[];
  visited: number;
}

export interface SearchTrace {
  /** ordered top level → 0 */
  layers: TraceLayer[];
  visitedTotal: number;
}

export interface LevelStats {
  level: number;
  nodes: number;
  avgDegree: number;
  maxDegree: number;
}

export interface IndexStats {
  count: number;
  deleted: number;
  maxLevel: number;
  levels: LevelStats[];
  params: {
    dim: number;
    M: number;
    M0: number;
    efConstruction: number;
    metric: Metric;
  };
}
