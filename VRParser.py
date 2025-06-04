class VRParser:
    def __init__(self, filepath, return_to_depot=True):
        self.filepath = filepath
        self.name = None
        self.type = None
        self.dimension = None
        self.edge_weight_type = None
        self.edge_weight_format = None
        self.capacity = None
        self.vehicles = None
        self.edge_weights = []
        self.demands = []
        self.depot = None
        self.return_to_depot = return_to_depot

        self.parse_file()

    def parse_file(self):
        with open(self.filepath, 'r') as file:
            section = None
            for line in file:
                line = line.strip()

                if line.startswith("NAME"):
                    self.name = line.split(':')[1].strip()
                elif line.startswith("TYPE"):
                    self.type = line.split(':')[1].strip()
                elif line.startswith("DIMENSION"):
                    self.dimension = int(line.split(':')[1].strip())
                elif line.startswith("EDGE_WEIGHT_TYPE"):
                    self.edge_weight_type = line.split(':')[1].strip()
                elif line.startswith("EDGE_WEIGHT_FORMAT"):
                    self.edge_weight_format = line.split(':')[1].strip()
                elif line.startswith("CAPACITY"):
                    self.capacity = int(line.split(':')[1].strip())
                elif line.startswith("VEHICLES") and not self.vehicles:
                    self.vehicles = int(line.split(':')[1].strip())
                elif line.startswith("EDGE_WEIGHT_SECTION"):
                    section = "EDGE_WEIGHT_SECTION"
                elif line.startswith("DEMAND_SECTION"):
                    section = "DEMAND_SECTION"
                elif line.startswith("DEPOT_SECTION"):
                    section = "DEPOT_SECTION"
                elif section == "EDGE_WEIGHT_SECTION" and line != "EOF":
                    self.edge_weights.append(list(map(int, line.split())))
                elif section == "DEMAND_SECTION" and line != "EOF":
                    node, demand = map(int, line.split())
                    self.demands.append((node, demand))
                elif section == "DEPOT_SECTION" and line != "EOF":
                    self.depot = int(line)

    def display_info(self):
        print(f"Problem Name: {self.name}")
        print(f"Problem Type: {self.type}")
        print(f"Number of Nodes (Dimension): {self.dimension}")
        print(f"Edge Weight Type: {self.edge_weight_type}")
        print(f"Edge Weight Format: {self.edge_weight_format}")
        print(f"Capacity: {self.capacity}")
        print(f"Vehicles: {self.vehicles}")
        print(f"Depot: {self.depot}")
        print(f"Return to Depot: {self.return_to_depot}")
        print(f"Number of Demands: {len(self.demands)}")
        print(f"Number of Edge Weights: {len(self.edge_weights)}")
