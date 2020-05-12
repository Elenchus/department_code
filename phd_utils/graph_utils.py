'''functions for quick graphing'''
from copy import deepcopy
from datetime import datetime
# from cuml import UMAP as umap
import holoviews as hv
import imgkit
import numpy as np
import networkx as nx
import pandas as pd
import pygraphviz as pgv
import random
from matplotlib import pyplot as plt
from nxviz.plots import CircosPlot
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from tqdm import tqdm

hv.extension('matplotlib')
pandas2ri.activate()

class GraphUtils():
    def __init__(self, logger):
        self.logger = logger

    def basic_histogram(self, data, filename):
        '''creates and saves a histogram'''
        self.logger.log("Plotting histogram")
        n_bins = len(set(data))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(data, n_bins)
        self.save_plt_fig(fig, filename)
        
    def bron_kerbosch(self, graph):
        def iterative_bk(P,R,X):
            if not any((P, X)):
                yield R
            
            try:
                u = random.choice(list(P.union(X)))
                S = P.difference(graph[u])
            # if union of P and X is empty
            except IndexError:
                S = P
            for v in S:
                yield from iterative_bk(P.intersection(graph[v]), R.union([v]), X.intersection(graph[v]))
                P.remove(v)
                X.add(v)

        P = self.flatten_graph_dict(graph)
        graph = self.convert_pgv_to_simple(graph)
        X = set()
        R = set()

        return list(iterative_bk(P,X,R))
       
    def categorical_plot_group(self, x, y, legend_labels, title, filename, axis_labels=None):
        '''creates and saves a categorical plot'''
        self.logger.log(f"Plotting bar chart: {title}")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        assert len(x) == len(y)
        for i in range(len(y)):
            ax.scatter([str(q) for q in x[i]], y[i], label=legend_labels[i])
    
        # plt.xticks(range(x[0]), (str(i) for i in x[0]))
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ttl = fig.suptitle(title)
        if axis_labels is not None:
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])

        self.save_plt_fig(fig, filename, (lgd, ttl, ))

    def contract_largest_maximum_cliques(self, graph):
        nodes = self.flatten_graph_dict(graph)
        maximum_cliques = self.bron_kerbosch(graph)
        clique_conversion = {}
        for node in nodes:
            max_size = 0
            max_id = None
            for i, clique in enumerate(maximum_cliques):
                if node in clique:
                    size = len(clique)
                    if size > max_size and size > 2:
                        max_size = size
                        max_id = i

            clique_conversion[node] = f"clique_{max_id}"

        converted_graph = {}
        for key in tqdm(graph):
            new_key = clique_conversion.get(key, None)
            if new_key is None:
                new_key = key

            converted_graph[new_key] = set()
            for node in graph[key]:
                new_node = clique_conversion.get(node, None)
                if new_node is None:
                    new_node = node

                converted_graph[new_key].add(new_node)

        converted_graph = self.convert_simple_to_pgv(converted_graph)

        return converted_graph

    def convert_adjacency_matrix_to_graph(self, a_m):
        items = a_m.columns
        graph = {}
        for ante in items:
            for con in items:
                val = a_m.at[ante, con]
                if val != 0:
                    if ante in graph:
                        graph[ante][con] = None
                    else:
                        graph[ante] = {con: None}

        return graph

    def convert_graph_to_adjacency_matrix(self, graph):
        items = [x for x in self.flatten_graph_dict(graph)]
        a_m = pd.DataFrame(0, index=items, columns=items)
        for ante in graph.keys():
            for con in graph[ante].keys():
                a_m.at[ante, con] += 1

        return a_m

    def convert_pgv_to_hv(self, graph):
        source = []
        target = []
        node_map = {}
        x = 0
        for s in graph:
            for t in graph[s]:
                if s not in node_map:
                    node_map[s] = x
                    x += 1
                
                if t not in node_map:
                    node_map[t] = x
                    x += 1

                s_n = node_map[s]
                t_n = node_map[t]
                source.append(s_n)
                target.append(t_n)

        node_name = []
        node_index = []
        for k, v in node_map.items():
            node_name.append(k)
            node_index.append(v)

        node_list = [x for _,x in sorted(zip(node_index, node_name))]

        return source, target

    def convert_pgv_to_simple(self, graph):
        ng = {}
        for k, v in graph.items():
            ng[k] = set(v.keys())

        return ng

    def convert_simple_to_pgv(self, graph):
        ng = {}
        for k, v in graph.items():
            ng[k] = {x: {} for x in v}

        return ng

    def create_boxplot(self, data, title, filename):
        '''creates and saves a single boxplot'''
        self.logger.log(f"Plotting boxplot: {title}")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(data)
        fig.suptitle(title)
        self.save_plt_fig(fig, filename)

    def create_boxplot_group(self, data, labels, title, filename, axis_labels=None):
        '''creates and saves a group of boxplots'''
        self.logger.log(f"Plotting boxplot group: {title}")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(data)
        fig.suptitle(title)
        ax.set_xticklabels(labels)
        if axis_labels is not None:
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])

        self.save_plt_fig(fig, filename)

    def create_feature_matrix_from_graph(self, graph):
        idx = self.flatten_graph_dict(graph)
        mat = pd.DataFrame(0, index=idx, columns=idx)
        for item in idx:
            mat.at[idx, idx] = 1

        return mat

    def create_grouped_barchart(self, data, bar_labels, group_labels, title, filename, axis_labels):
        bar_width = 0.25
        r = []
        r.append(np.arange(len(data[0])))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(0, len(data)):
            ax.bar(r[-1], data[i], width=bar_width, label=bar_labels[i])
            r.append([x + bar_width for x in r[-1]])
        ax.set_xticks([r + bar_width for r in range(len(group_labels))])
        ax.set_xticklabels(group_labels, rotation=45, ha="right")
        if axis_labels is not None:
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])

        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ttl = fig.suptitle(title)
        self.save_plt_fig(fig, filename, (lgd, ttl, ))

    def create_hv_chord(self, source, target):
        # dset = hv.Dataset(pd.DataFrame(node_list), 'index')
        chord = hv.Chord(((source,target),))
        # chord.opts(hv.opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=hv.dim('source').str(), 
        #        labels='name', node_color=hv.dim('index').str()))

        return chord

    def create_hv_graph(self, source, target):
        graph = hv.Graph(((source,target),))
        graph.opts(xaxis=None,yaxis=None, directed=True,arrowhead_length=0.1)
        graph = hv.element.graphs.layout_nodes(graph, layout=nx.drawing.layout.shell_layout) 

        return graph
    
    def create_scatter_plot(self, data, labels, title, filename, legend_names=None):
        '''creates and saves a scatter plot'''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels)
        if legend_names is None:
            legend = ax.legend(*scatter.legend_elements(), \
                        loc="upper left", title="Cluster no.", bbox_to_anchor=(1, 0.5))
        else:
            handles, _ = scatter.legend_elements(num=None)
            legend = ax.legend(handles, legend_names, loc="upper left", title="Legend", bbox_to_anchor=(1, 0.5))

        ttl = fig.suptitle(title)

        filename = self.logger.output_path / filename

        self.save_plt_fig(fig, filename, [ttl, legend])

    def create_visnetwork(self, graph, name, title, attrs=None):
        if attrs:
            nodes = pd.DataFrame(attrs)
            nodes = nodes.transpose()
            nodes['id'] = nodes.index
            nodes['label'] = nodes.index
        else:
            nodes = pd.DataFrame(self.flatten_graph_dict(graph), columns=['id'])

        from_nodes = []
        to_nodes = []
        colors = []
        for key, val in graph.items():
            for k in val.keys():
                from_nodes.append(key)
                to_nodes.append(k)
                color = graph[key][k].get('color', 'black')
                colors.append(color)

        edges = pd.DataFrame([from_nodes, to_nodes, colors])
        edges = edges.transpose()
        edges.columns=['from', 'to', 'color']

        visnet = importr('visNetwork')
        r_nodes = pandas2ri.py2ri(nodes)
        r_edges = pandas2ri.py2ri(edges)

        net = visnet.visNetwork(r_nodes, r_edges, main = title, width = "100%", improvedLayout=False)
        net = visnet.visEdges(net, arrows = 'to') 
        net = visnet.visNodes(net, shape='circle')
        vispath = self.logger.output_path / f"{name}"
        vishtml = f"{vispath}.html"
        visnet.visSave(net, vishtml)
        vispng = f"{vispath}.png"
        imgkit.from_file(vishtml, vispng)

    def flatten_graph_dict(self, dictionary):
        temp = set()
        for k, v in dictionary.items():
            temp.add(k)
            for key in v.keys():
                temp.add(key)

        return temp

    def graph_component_finder(self, graph):
        am = self.convert_graph_to_adjacency_matrix(graph)
        keys = am.index.tolist()
        ITEMS = am.index.tolist()
        def search_matrix(node, current_component, connected_nodes):
            for item in ITEMS:
                row = am.at[node, item]
                col = am.at[item, node]
                if (row > 0 or col > 0) and item not in current_component:
                    current_component.add(item)
                    connected_nodes.add(item)

        identified = set()
        components = []       
        while len(keys) > 0:
            component = set()
            connected_nodes = set()
            node = keys.pop()
            if node in identified:
                continue

            component.add(node)
            while True:
                search_matrix(node, component, connected_nodes)
                if not connected_nodes:
                    break

                node = connected_nodes.pop()
            
            identified.update(component)
            components.append(component)

        return components
                
    def graph_edit_distance(self, expected, test, attrs=None):
        if not test:
            return 0, {}, {}

        expected = self.stringify_graph(expected)
        test = self.stringify_graph(test)

        ged_score = 0
        d = {x: {} for x in self.flatten_graph_dict(test)}
        if attrs == None:
            attrs = {x: {} for x in d}

        possible_nodes = list(x for x in self.flatten_graph_dict(expected))
        keys = list(d.keys())
        edit_attrs = {}
        edit_history = deepcopy(test)
        for key in keys:
            if key not in possible_nodes:
                if key in attrs:
                    ged_score += attrs[key].get('weight', 1) # fee
                else:
                    ged_score += 1

                if key in test:
                    edges = test[key]
                    # ged_score += sum([test[key][x].get('weight', 1) for x in edges]) # confidence
                    for k in edges:
                        edit_history[key][k]['color'] = 'red'

                edit_attrs[key] = {'shape': 'house'}
                d.pop(key)

        nodes_to_add = set()
        edges_to_add = {}
        for key in d.keys():
            if key not in expected:
                continue

            possible_edges = set(x for x in expected[key].keys())
            if key in test:
                actual_edges = set(test[key].keys())
            else:
                actual_edges = set()

            missing_edges = possible_edges - actual_edges
            should_have = possible_edges.intersection(actual_edges)
            should_have.update(missing_edges)
            should_not_have = actual_edges - possible_edges
            d[key] = {x: {} for x in should_have}
            missing_nodes = set()
            for node in missing_edges:
                if node not in d:
                    missing_nodes.add(node)
            
            edges_to_add[key] = missing_edges

            nodes_to_add.update(missing_nodes)
            # ged_score += sum([expected[key][x].get('weight', 1) for x in missing_edges]) + sum([test[key][x].get('weight', 1) for x in should_not_have]) # confidence
            for k in should_not_have:
                edit_history[key][k]['color'] = 'red'

        for key in edges_to_add:
            for k in edges_to_add[key]:
                if key not in edit_history:
                    edit_history[key] = {}

                edit_history[key][k] = {'color': 'blue'}
                
        while True:# infinite loop!
            if len(nodes_to_add) == 0:
                break

            ignore_list = []
            node = nodes_to_add.pop()
            if node in attrs:
                ged_score += attrs[node].get('weight', 1) # fee
            else:
                ged_score += 1

            edit_attrs[node] = {'shape': 'invhouse'}

            if node not in expected:
                ignore_list.append(node)
                continue


            edges = expected[node]
            # ged_score += sum([expected[node][x].get('weight', 1) for x in edges]) # confidence
            d[node] = edges
            edit_history[node] = edges
            for k in edit_history[node]:
                edit_history[node][k]['color'] = 'blue'

            for new_node in edges:
                if new_node not in d and new_node not in ignore_list:
                    nodes_to_add.add(new_node)

        return ged_score, edit_history, edit_attrs

    def plot_circos_graph(self, graph, attrs, filename):
        G = self.convert_pgv_to_simple(graph)
        formatted = nx.DiGraph(G)
        if attrs is not None:
            for node in attrs:
                formatted.node[node]["color"] = attrs[node]['color']

        c = CircosPlot(formatted, node_labels=True, node_color="color")

        c.draw()
        plt.savefig(filename)

    def plot_tsne(self, new_values, labels, title):
        '''Create and save a t-SNE plot'''
        self.logger.log(f"Plotting TSNE figure")
        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
    
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111)
        for i in range(len(x)):
            ax.scatter(x[i], y[i])

        for i in range(len(x)):
            ax.annotate(labels[i],
                        xy=(x[i], y[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')

        fig.suptitle(title)
    
        name = "t-SNE_" + datetime.now().strftime("%Y%m%dT%H%M%S")
        path = self.logger.output_path / name
        self.logger.log(f"Saving TSNE figure to {path}")
        fig.savefig(path)

    def save_hv_fig(self, fig, filename):
        current = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_path = f"{filename}_{current}.png"
        if self.logger is not None:
            output_path = self.logger.output_path / output_path

        hv.save(fig, output_path)

    def save_plt_fig(self, fig, filename, bbox_extra_artists=None):
        '''Save a plot figure to file with timestamp'''
        current = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_path = f"{filename}_{current}"
        if self.logger is not None:
            output_path = self.logger.output_path / output_path

        if bbox_extra_artists is None:
            fig.savefig(output_path)
        else:
            fig.savefig(output_path, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')

        plt.close(fig)

    def stringify_graph(self, graph):
        str_graph = {}
        for key in graph:
            str_graph[str(key)] = {}
            for k in graph[key]:
                str_graph[str(key)][str(k)] = graph[key][k]

        return str_graph

    def plot_umap(self, embedding, title):
        '''Create and save a UMAP plot'''
        self.logger.log("Plotting UMAP")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral')
        # ax.gca().set_aspect('equal', 'datalim')
        fig.suptitle(title)

        name = "UMAP_" + datetime.now().strftime("%Y%m%dT%H%M%S")
        path = self.logger.output_path / name
        fig.savefig(path)

    def visual_graph(self, data_dict, output_file, title=None, directed=True, node_attrs=None, graph_style='fdp'):
        max_len = 0
        # sub_list_of_lists = [data_dict[key].keys() for key in data_dict.keys()]
        # full_list = set(list(str(x) for x in data_dict.keys()) + list(str(item) for elem in sub_list_of_lists for item in elem))
        full_list = self.flatten_graph_dict(data_dict)
        for s in full_list:
            if len(s) > max_len:
                max_len = len(s)

        if max_len < 10:
            width = 2
        else:
            width = 5

        A = pgv.AGraph(data=data_dict, directed=directed)
        if title is not None:
            A.graph_attr['fontsize'] = 30
            A.graph_attr['label'] = title
            A.graph_attr['labelloc'] = 't'

        A.node_attr['style']='filled'
        A.node_attr['shape'] = 'circle'
        A.node_attr['fixedsize']='true'
        A.node_attr['fontsize'] = 25
        A.node_attr['height']=width 
        A.node_attr['width']=width
        A.node_attr['fontcolor']='#000000'
        A.edge_attr['penwidth']=7
        if directed:
            A.edge_attr['style']='tapered'
        else:
            A.edge_attr['style']='solid'

        for k, v in data_dict.items():
            for node, d in v.items():
                if d is not None:
                    edge = A.get_edge(k, node)
                    for att, val in d.items():
                        edge.attr[att] = val
                        True
        
        if node_attrs is not None:
            for k, v in node_attrs.items():
                node = A.get_node(k)
                for attr, val in v.items():
                    node.attr[attr] = val


        A.draw(str(output_file), prog=graph_style)

    def graph_legend(self, data_dict, output_file, title=None):
        max_len = 0
        full_list = self.flatten_graph_dict(data_dict)
        for s in full_list:
            if len(s) > max_len:
                max_len = len(s)

        if max_len < 10:
            width = 2
        else:
            width = 5

        A = pgv.AGraph(data={})
        if title is not None:
            A.graph_attr['fontsize'] = 15
            A.graph_attr['label'] = title
            A.graph_attr['labelloc'] = 't'

        A.node_attr['style']='filled'
        A.node_attr['shape'] = 'circle'
        A.node_attr['fixedsize']='true'
        A.node_attr['height']=width 
        A.node_attr['width']=width
        A.node_attr['fontcolor']='#000000'
        A.edge_attr['penwidth']=7
        A.edge_attr['style']='invis'

        nbunch = list(data_dict.keys())
        for i, node in enumerate(nbunch):
            A.add_node(node)
            n = A.get_node(node)
            n.attr['shape'] = 'rectangle'
            n.attr['rank'] = 'max'
            n.attr['fontsize'] = 15
            for attr, val in data_dict[node].items():
                n.attr[attr] = val

            if i < len(nbunch) - 1:
                A.add_edge(node, nbunch[i+1])# , style='invis')

        A.add_subgraph(nbunch=nbunch, name='Legend')
        l = A.get_subgraph('Legend')
        l.rank = 'max'
        l.label = 'Legend'
        l.style = 'filled'
        l.shape = 'rectangle'
        l.labelloc = 't'
        l.fontcolor = '#000000'
        l.color = 'grey'
        l.pack = True

        A.draw(str(output_file), prog='dot')