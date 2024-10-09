from ortools.sat.python import cp_model
import numpy as np
import os
import json

'''
To implement:
Implement Solution for the roster to be compatible with previous months roster
'''

class Solver():
    '''
    *IMPORTANT: Days indexing start with 0

    An optimization solver for scheduling employees based on roles and role requirements.
    This class provides a framework for solving workforce scheduling problems using Google CP Solver. It handles the creation of the optimization model, including binary decision variables, constraints, and objective functions. Child classes can inherit from this class and override specific methods to implement custom constraints for their unique problem domains.

    Variable Name format: 'resource_n1r1d1s1'
    self.numResources       : Number of resources/Employees
    self.numRoles           : Total Number of Roles for all resources
    self.resourceIdLookup   : Dictionary {resourceName: allocatedResourceId}
    self.resourceLookup     : Dictionary {allocatedResourceId: resourceName}
    self.rolesIdLookup      : Dictionary {role:allocatedRoleId}
    self.rolesLookup        : Dictionary {allocatedRoleId:role}
    self.resourceRoleMapping: Dictionary {Resourceid: List[Roles]}

    Functions:
    _createModel
    _createVars
    _createEmployeeRoleLookup
    _removeExtraRoles
    _createRequirementMapping
    _setRequirementConstraint
    _maxOneShiftPerDay
    _nextDayShift
    _banShift
    _childConstraints
    _finalSolve
    solve
    _maxHoursPerWeek
    getRosterByResource
    getRosterByDay
    getRosterAllResources
    getRosterAllDays
    '''
    def __init__(self, maxSearchTime = 250, systemCore = True):
        self.maxSearchTime = maxSearchTime
        if systemCore:
            self.numCores = os.cpu_count()
        else:
            self.numCores = 1

    def _createModel(self):
        """
        Creates a new constraint programming model using the cp_model library.
        This model will be used to formulate and solve the optimization problem.
        **Attributes:**
            self.model (CpModel): The newly created constraint programming model.
        """
        self.model = cp_model.CpModel()
    
    def _createVars(self):
        """
        Creates individual binary variables for all employees with all roles for all days and shifts.
        Each variable represents whether or not an employee is assigned to a particular role on a specific day and shift.
        Variable names follow a consistent format: 
            'resource_n1r1d1s1', where
            'n' represents the employee index,
            'r' represents the role index, 
            'd' represents the day index, 
            's' represents the shift index.

        **Attributes:**
            self.vars (np.ndarray<object>): A four-dimensional NumPy array of binary variables, where each variable represents whether or not an employee is assigned to a particular role on a specific day and shift.
            Usage: self.vars[resourceID][roleID][Day][Shift]
         """
        self.vars = np.empty(shape=(self.numResources, self.numRoles, self.numDays, self.numShifts), dtype='object')
        for n in range(self.numResources):
            for r in range(self.numRoles):
                for d in range(self.numDays):
                    for s in range(self.numShifts):
                        self.vars[n][r][d][s] = self.model.NewBoolVar(f'resource_n{n}r{r}d{d}s{s}')

    def _createEmployeeRoleLookup(self,arg):
        """
        Creates lookup tables for employees, roles, and their assigned roles.

        **Parameters:**
            arg (dict): Dictionary with resource keys (names or IDs) and a list of associated roles for each resource.

        **Attributes:**
            self.numResources (int): Number of resources/employees.
            self.numRoles (int): Total number of roles across all resources.
            self.resourceIdLookup (dict): Dictionary mapping resource names or IDs to their allocated resource IDs.
            self.resourceLookup (dict): Dictionary mapping allocated resource IDs to resource names or IDs.
            self.rolesIdLookup (dict): Dictionary mapping role names to their allocated role IDs.
            self.rolesLookup (dict): Dictionary mapping allocated role IDs to role names.
            self.resourceRoleMapping (dict): Dictionary mapping resource IDs to a list of their assigned role IDs.
        **Example Usage:**
            ```python
            employee_role_dict = {
                "Employee1": ["Role1", "Role3"],
                "Employee2": ["Role2", "Role4"],
            }

            self._createEmployeeRoleLookup(employee_role_dict)

            # Access the lookup tables:
            number_of_employees = self.numResources
            number_of_roles = self.numRoles
            resourceId = self.resourceIdLookup[resourceName]
            resource = self.resourceLookup[resourceId]
            roleId = self.rolesIdLookup[roleName]
            role = self.rolesLookup[roleId]
            RolesAssigned = self.resourceRoleMapping[resourceId]
            ```

        **Internal Details:**

            1. Creates resource and role lookup tables:
                - `resourceIdLookup`: Maps resource names or IDs to allocated resource IDs.
                - `resourceLookup`: Maps allocated resource IDs to resource names or IDs.
                - `rolesIdLookup`: Maps role names to allocated role IDs.
                - `rolesLookup`: Maps allocated role IDs to role names.

            2. Creates a resource-role mapping table:
                - `resourceRoleMapping`: Maps resource IDs to a list of their assigned role IDs.
        """

        #Creating Resource Lookup
        ids = [resource for resource in arg]
        resources = []
        for resource in ids:
            if not (resource in resources):
                resources.append(resource)

        self.resourcesLookup = {id:resource for id, resource in enumerate(resources)}
        self.resourceIdLookup = {resource:id for id, resource in enumerate(resources)}
        self.numResources = len(resources)

        #Create Roles Lookup
        ids = [arg[resource] for resource in arg]
        roles = []
        for roleList in ids:
            for role in roleList:
                if not (role in roles):
                    roles.append(role)
        self.rolesLookup = {id:role for id, role in enumerate(roles)}
        self.rolesIdLookup = {role:id for id, role in enumerate(roles)}
        self.numRoles = len(roles) 

        #Create Resource Role mapping from original args
        self.resourceRoleMapping = {}
        for resource in arg:
            roleList = arg[resource]
            roleIdList = [self.rolesIdLookup[role] for role in roleList]
            self.resourceRoleMapping[self.resourceIdLookup[resource]] = roleIdList

    def _removeExtraRoles(self):
        """
        Removes extra roles that were not assigned to any resource during variable creation.
        This ensures that the optimization problem only considers roles that are actually relevant for the scheduling problem.
        **Internal Details:**
            1. Identifies roles that are not assigned to any resource:
                For each resource, check if all roles exist in the resource-role mapping. If a role is missing, it's considered an extra role.
            2. Enforces constraints to prevent assigning extra roles:
                For each extra role identified for a resource, add constraints to the model that set the corresponding variable to zero. This ensures that the optimizer cannot assign the extra role to the resource.
        """
        for resource in range(self.numResources):
            roleAbsent = [role for role in range(self.numRoles) if not role in self.resourceRoleMapping[resource]]
            for role in roleAbsent:
                for d in range(self.numDays):
                    for s in range(self.numShifts):
                        self.model.Add(self.vars[resource][role][d][s] == 0)

    def _createRequirementMapping(self,arg):
        """
        Creates a mapping of daily role requirements for each shift.
        
        **Parameters:**
            arg (list[list[dict]]): A two-dimensional matrix of role requirement dictionaries, where each row represents a day and each column represents a shift. Each role requirement dictionary has the following structure:
                `{roleID: requirement}`
                where `roleID` is the ID of the role and `requirement` is the number of employees required for that role in that shift.

        **Attributes:**
            self.requirementMapping (list[list[dict]]): A two-dimensional list representing the mapping of daily role requirements for each shift.

        **Example Usage:**
            ```python
            role_requirements = [
                [
                    {"Role1": 2, "Role2": 1},
                    {"Role1": 3, "Role2": 2},
                    {"Role1": 1, "Role2": 1},
                ],
                [
                    {"Role1": 0, "Role2": 0},
                    {"Role1": 1, "Role2": 0},
                    {"Role1": 0, "Role2": 0},
                ],
            ]
            self._createRequirementMapping(role_requirements)
        
            Use:
            requirement = self.requirementMapping[day][shift][roleID]

        **Internal Details:**
            1. Initializes an empty list to store daily role requirements
            2. Iterates through each day's role requirements
                    - Creates an empty list to store shift-wise role requirements
                    - Iterates through each shift's role requirements
                            - Creates a dictionary to map role IDs to their respective requirements
                    - Appends the day's role requirements to the main list
        """
        self.requirementMapping = []
        for day in arg:
            dayRequirements = []
            for shift in day:
                dayRequirements.append({self.rolesIdLookup[role]:shift[role] for role in shift})
            self.requirementMapping.append(dayRequirements)

    def _setRequirementConstraint(self):
        """
        Sets constraints to ensure that the number of employees assigned to each role for a given day and shift meets the specified requirements.

        **Internal Details:**
            1. Iterates through each day and shift:
            2. Retrieves the role requirements for the current day and shift
            3. Iterates through each role and its corresponding requirement
            4. The constraint ensures that the sum of the binary variables `self.vars[resource][role][d][s]` across all resources is greater than or equal to the required number of employees for that role.
        """
        for d in range(self.numDays):
            for s in range(self.numShifts):
                roleRequirements = self.requirementMapping[d][s]
                for role in roleRequirements:
                    self.model.Add(sum(self.vars[resource][role][d][s] for resource in range(self.numResources)) >= roleRequirements[role])
                
    def _maxOneShiftPerDay(self):
        """
        Ensures that each resource is assigned to at most one shift per day.
        This constraint prevents over-scheduling resources and ensures that each resource can only work one shift per day.

        **Internal Details:**
            1. Iterates through each resource and day
            2. For each resource and day, constructs a constraint that sums the binary variables `self.vars[n][r][d][s]` across all roles and shifts
            3. The constraint ensures that the sum of the binary variables representing the assigned shifts for a resource on a particular day is less than or equal to one. This effectively limits the resource to at most one shift per day.
        """
        for n in range(self.numResources):
            for d in range(self.numDays):
                self.model.Add(sum(self.vars[n][r][d][s] for r in range(self.numRoles) for s in range(self.numShifts)) <= 1)

    def _nextDayShift(self,arg):
        """
        Enforces constraints to prevent assigning incompatible shift combinations for consecutive days.

        **Parameters:**
            arg (list[dict]): A list of dictionaries, each representing a non-sequential shift combination. Each dictionary has the following structure:
                `{origin: Shiftid, destination: Shiftid}`
                where `origin` is the shift ID of the origin shift and `destination` is the shift ID of the destination shift.

        **Internal Details:**
            1. Initializes a NumPy array to store shift compatibility information
            2. If the `arg` list is not empty, process the non-sequential shift combinations:
                        - Extract the origin and destination shift indices
                        - Set the corresponding element in the compatibility matrix to indicate the incompatibility
            3. Iterates through each resource, day, and shift:
                        - Constructs a constraint that considers all possible shift combinations for the next day
                        - The constraint ensures that the sum of the binary variables representing the assigned shifts for the next day is less than or equal to one. This effectively prevents assigning incompatible shift combinations for consecutive days.
        """
        self.non_sequential_shifts_indices = np.zeros(shape=(self.numShifts, self.numShifts), dtype='object')
        if arg:
            for dependence in arg:
                i_idx = dependence[0]
                j_idx = dependence[0]
                self.non_sequential_shifts_indices[i_idx][j_idx] = 1

        for n in range(self.numResources):
            for d in range(self.numDays - 1):
                for s in range(self.numShifts):
                    self.model.Add(
                        sum(self.vars[n][r][d][s] * self.non_sequential_shifts_indices[s][j] +
                            self.vars[n][r][d + 1][j]
                            for j in range(self.numShifts) for r in range(self.numRoles)) <= 1)
                    
    def _banShift(self, arg):
        """
        Prevents a specific resource from being assigned to a particular shift on a given day.
        This constraint enforces a hard leave rule, ensuring that the resource is not scheduled for the specified shift.
        **Parameters:**
            arg (tuple): A tuple containing the (resource ID, day ID, and shift ID).

        **Internal Details:**
            1. Extracts the resource ID, day ID, and shift ID from the `arg` tuple
            2. Iterates through all roles for the specified resource
            3. For each role, adds a constraint that sets the corresponding binary variable to zero
            4. This constraint ensures that the binary variable representing the assignment of the specified resource to the specified shift on the specified day is always zero, effectively banning that shift for the resource.
        """
        for r in range(self.numRoles):
            self.model.Add(self.vars[arg[0]][r][arg[1]][arg[2]] == 0)
    
    def _maxHoursPerWeek(self, weekStartDay, shiftHoursMapping):
        pass
    
    def childConstraints(self,args):
        """
        This function is a placeholder for child classes to implement additional constraints specific to their problem domain.
        **Parameters:**
        args (list): A list containing the problem arguments:
            - `args[0]` (dict): A dictionary with employee keys (names or IDs) and a list of associated roles for each employee.
            - `args[1]` (list[list[dict]]): A two-dimensional matrix of role requirement dictionaries, where each row represents a day and each column represents a shift. Each role requirement dictionary has the following structure:
                `{roleID: requirement}`
                where `roleID` is the ID of the role and `requirement` is the number of employees required for that role in that shift.

        **Internal Details:**
            This function is intended to be overridden by child classes to implement additional constraints specific to their problem domain. The base class provides the necessary framework for creating and solving the optimization problem using Google CP Solver, and child classes can extend this functionality by adding their own constraints.
            Child classes should implement their specific constraints within this function, utilizing the `self.model` object to add constraints to the optimization problem. The `self.model` object is a CP-SAT constraint satisfaction problem solver that provides various methods for defining constraints.
            Once the child class has implemented its specific constraints, it should call the `_finalSolve` function to solve the optimization problem using Google CP Solver. The `_finalSolve` function will handle the actual solving process and return the status code of the solution.
        """
        pass

    def _finalSolve(self):
        """
        Solves the optimization problem using Google CP Solver.
        **Attributes:**
            self.solver (cp_model.CpSolver): The Google CP Solver object used to solve the optimization problem.
            self.maxSearchTime (float): The maximum time in seconds allowed for solving the optimization problem.
            self.numCores (int): The number of CPU cores to use for parallel solving.
        **Returns:**
            status (cp_model.CpSolverStatus): The status code of the optimization solution.
        **Internal Details:**
            1. Creates a Google CP Solver object
            2. Sets the maximum time in seconds allowed for solving the optimization problem
            3. Sets the number of CPU cores to use for parallel solving
            4. Solves the optimization problem using the CP Solver
            5. Returns the status code of the optimization solution
        The optimization solution can be accessed using the `self.solver` object.
        """
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = self.maxSearchTime
        self.solver.num_search_workers = self.numCores
        status = self.solver.Solve(self.model)
        return status

    def solve(self,args):
        """
        Solves the scheduling optimization problem using the specified employee role mapping and role requirement data.
        The optimization solution can be accessed using the `self.solver` object.
        **Parameters:**
            args (list): A list containing the problem arguments:
                - `args[0]` (dict): A dictionary with employee keys (names or IDs) and a list of associated roles for each employee.
                - `args[1]` (list[list[dict]]): A two-dimensional matrix of role requirement dictionaries, where each row represents a day and each column represents a shift. Each role requirement dictionary has the following structure:
                    `{roleID: requirement}`
                    where `roleID` is the ID of the role and `requirement` is the number of employees required for that role in that shift.

        **Returns:**
            cp_model.CpSolverStatus: The status code of the optimization solution. The solution can be accessed using the `self.solver` object.

        **Internal Details:**
            1. Creates the optimization model
            2. Sets the number of days and shifts
            3. Creates employee-role lookup tables
            4. Creates binary decision variables
            5. Removes extra roles that were not assigned to any resource during variable creation:
            6. Creates a mapping of daily role requirements for each shift
            7. Sets constraints to ensure that the number of employees assigned to each role for a given day and shift meets the specified requirements
            8. Enforces constraints to prevent assigning incompatible shift combinations for consecutive days
            9. Enforces constraints specific to the child class's problem domain
            10. Solves the optimization problem using Google CP Solver
            11. Returns the status code of the optimization solution
        """

        self._createModel()
        self.numDays = len(args[1])
        self.numShifts = len(args[1][0])
        self._createEmployeeRoleLookup(args[0])
        self._createVars()
        self._removeExtraRoles()
        self._createRequirementMapping(args[1])
        self._setRequirementConstraint()
        self._maxOneShiftPerDay()
        self.childConstraints(args)

        self._finalSolve()

    def getRosterByResource(self,resourceName):
        resourceID = self.resourceIdLookup[resourceName]
        resourceCalendar = {'Resource':resourceName}
        for d in range(self.numDays):
            day = {}
            for s in range(self.numShifts):
                for r in range(self.numRoles):
                    if self.solver.Value(self.vars[resourceID][r][d][s]):
                        day['Role'] = self.rolesLookup[r]
                        day['Shift'] = s
                        resourceCalendar[f'Day_{d}'] = day
        return resourceCalendar
    
    def getRosterAllResources(self):
        calendar = {}
        for n in range(self.numResources):
            calendar[self.resourcesLookup[n]] = self.getRosterByResource(self.resourcesLookup[n])
        return calendar
    
    def getRosterByDay(self,day):
        dayCalendar = {'Day':day}
        for s in range(self.numShifts):
            shift = {'Shift':s}
            for r in range(self.numRoles):
                role = {'Resources':[]}
                for n in range(self.numResources):
                    if self.solver.Value(self.vars[n][r][day][s]):
                        role['Requirement'] = self.requirementMapping[day][s][r],
                        role['Resources'].append(self.resourcesLookup[n])
                if len(role['Resources']) != 0:
                    shift[self.rolesLookup[r]] = role
            if len(shift) >= 1:
                dayCalendar[f'Shift {s}'] = shift
        return dayCalendar
    
    def getRosterAllDays(self):
        Calendar = {}
        for d in range(self.numDays):
            Calendar[f'Day {d}'] = self.getRosterByDay(d)

        return Calendar

class HospitalSolver(Solver):
    def __init__(self):
        '''
        *IMPORTANT: Days indexing start with 0
        args[0]: Dictionary with employee Key (Name or Id) with a list of Roles Key
        args[1]: 2D matrix of Role Requirement Dictionary with each row defining a day and each column defining a shift
        args[2]: List of (ResourceName ,Leave Date) -- Hard Rule
        args[3]: List of (ResourceName ,Leave Date, Priority) -- Soft Rule
        '''
        super().__init__() 
        self.numShifts = 3

    def _dayAfterNight(self):
        '''
        Grants holiday after night shift
        '''
        self._nextDayShift([(2,0),(2,1),(2,2)])

    def _hardLeave(self,arg):
        '''
        Grants compulsory leave
        '''
        for leave in arg:
            for shift in range(self.numShifts):
                self._banShift((self.resourceIdLookup[leave[0]],leave[1],shift))
    
    def _softLeave(self,arg):
        resourcesLeaveRequests = np.zeros(shape=(self.numResources, self.numRoles,self.numDays, self.numShifts), dtype='object')
        for request in arg:
            for s in range(self.numShifts):
                for r in range(self.numRoles):
                    resourcesLeaveRequests[self.resourceIdLookup[request[0]]][r][request[1]][s] = request[2]
        
        self.model.Minimize(sum(resourcesLeaveRequests[n][r][d][s]*self.vars[n][r][d][s]
                            for n in range(self.numResources)
                            for r in range(self.numRoles)
                            for d in range(self.numDays)
                            for s in range(self.numShifts)))


    def childConstraints(self,args):
        '''
        args[0]: Dictionary with employee Key (Name or Id) with a list of Roles Key
        args[1]: 2D matrix of Role Requirement Dictionary with each row defining a day and each column defining a shift
        args[2]: List of (ResourceName ,Leave Date) -- Hard Rule
        args[3]: List of (ResourceName ,Leave Date, Priority) -- Soft Rule
        '''
        self._dayAfterNight()
        self._hardLeave(args[2])
        self._softLeave(args[3])


if __name__=='__main__':
    solution = HospitalSolver()
    args = [
            {
             'Resource1':['Role1','Role2','Role3','Role4'],
             'Resource2':['Role1','Role2','Role3','Role4'],
             'Resource3':['Role1','Role2'],
             'Resource4':['Role1','Role2'],
             'Resource5':['Role1','Role2'],
             'Resource6':['Role1','Role2'],
             'Resource7':['Role1','Role2'],
             'Resource8':['Role1','Role2'],
             'Resource9':['Role1','Role2'],
             'Resource10':['Role1','Role2'],
             'Resource11':['Role1','Role2'],
             'Resource12':['Role1','Role2'],
             'Resource13':['Role1','Role2'],
             'Resource14':['Role1','Role2'],
             'Resource15':['Role1','Role2'],
             'Resource16':['Role1','Role2']
             },
             [
              [{'Role1':3, 'Role2':1},{'Role1':3},{'Role1':1}],
              [{'Role1':3},{'Role1':3},{'Role1':3}],
              [{'Role1':3},{'Role1':3},{'Role1':3}],
              [{'Role1':3},{'Role1':3},{'Role1':3}],
              [{'Role1':3},{'Role1':3},{'Role1':3}],
              [{'Role1':3},{'Role1':3},{'Role1':3}],
              [{'Role1':3},{'Role1':3},{'Role1':3}]
              ],
              [
                  ('Resource1',2),
                  ('Resource10',4)
              ],
              [
                  ('Resource4',3,0.8),
                  ('Resource3',4,0.7),
                  ('Resource2',4,0.6),
                  ('Resource1',4,0.5),
                  ('Resource3',2,0.5),
                  ('Resource4',4,0.2),
                  ('Resource5',4,0.3),
                  ('Resource6',4,0.4),
                  ('Resource7',4,0.7),
                  ('Resource8',4,0.7),
                  ('Resource9',4,0.3),
                  ('Resource10',4,0.7),
                  ('Resource11',4,0.7),
                  ('Resource12',4,0.7),
                  ('Resource13',4,0.7),
                  ('Resource14',4,0.7),
                  ('Resource15',4,0.7),
                  ('Resource16',4,0.7)

              ]
             ]
    solution.solve(args)
    with open('ResourceRoster','w') as f:
        json.dump(solution.getRosterAllResources(),f, indent=4)
    
    with open('DayRoster','w') as f:
        json.dump(solution.getRosterAllDays(),f, indent=4)
    print('Success')