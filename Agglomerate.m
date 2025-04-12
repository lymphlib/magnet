%> @file  Agglomerate.m
%> @author Andrea Re Fraschini
%> @date 20 may 2024
%> @brief Agglomerate mesh.
%>
%==========================================================================
%> @section class Class description
%==========================================================================
%> @brief Agglomerate the input mesh and convert it to lymph format. See
%> the `magnet` documentation for further details on the parameters.
%>
%> @param mesh_path   File path of the mesh to be agglomerated.
%> @param output_path File path where the final lymph mesh is saved.
%> @param Data        Struct with problem's data
%> @param SimType     String simulation type, used for boundary tag.
%> @param model       {'METIS','KMEANS','SageBase2D'}
%>                    Agglomeration model to be used.
%> @param mode        {'Nref','mult_factor'} Agglomeration mode.
%> @param param       Number of refinements or multiplicative factor,
%>                    depending on the agglomeration mode.
%> @param b_tags      tag(s) of boundary elements (optional).
%> @param b_tag_name  name of the field of boundary tags(optional).
%>
%> @retval region     Boundary connectivity and boundary tags.
%> @retval neighbor   Neigbor struct having fields:
%>                     - nedges(i) number of edges for element i.
%>                     - neigh{i}(j) el-id for neigh. of el. i edge j
%>                     - neighedges{i}(j) edge-id for neigh. of el. i edge j 
%==========================================================================
function [region, neighbor] = Agglomerate(mesh_path, ...
                                          output_path, ...
                                          Data, SimType, ...
                                          model, mode, ...
                                          nref, mult_factor, k, ...
					                      b_tags, b_tag_name)
    arguments
        mesh_path string
        output_path string
        Data struct
        SimType string

        model string = 'KMEANS'
        mode string = 'Nref'
        nref double = 7 
        mult_factor double = 0.1 
        k double = 128 
	    b_tags = string(missing)  % will become 'None' in Python
	    b_tag_name = string(missing)
    end
    
    input = strcat("magnet/lymphcomm.py ", ...
                    "--meshpath ", mesh_path, ...
                    " --aggmodel ", model, ...
                    " --mode ", mode, ...
                    " --nref ", string(nref), ...
                    " --multfactor ", string(mult_factor), ...
                    " --k ", string(k), ... % " --getboundary ", string(1), ...
                    " --tolymph ", string(1) ...
                    );
    tic
    %% set the Python virtual environment to use:
    
    fprintf('Setting up Python environment ... ')
    pyenv('Version', ... 
                'C:\Users\Andrea\Agglomeration\Scripts\python', ... 
                'ExecutionMode','OutOfProcess');
    fprintf('done\n')

    %% Run agglomeration script
    
    fprintf('Agglomerating the mesh ... ')
    [vertices, connectivity, physical_groups, boundary, b_tags] = pyrunfile( ...
        input, ...
        ["vertices" "connectivity" "physical_groups" "boundary" "b_tags"]); ...
    %     mesh_path=mesh_path, ... 
    %     agg_model=model, ...
    %     agg_mode=mode, ...
    %     agg_parameter=param, ...
	% b_tags=b_tags, ...
	% b_tag_name=b_tag_name ...
        
     fprintf('done\n')

    %% convert output to Lymph region:

    fprintf('Converting mesh to Lymph format ... ')
    
    % create region.connectivity by converting python output:
    region.connectivity = cell(connectivity);
    for jj= 1: length(region.connectivity)
        region.connectivity{jj} = int64(region.connectivity{jj});
    end
    
    % Create region.ne
    region.ne = length(region.connectivity) ;

    % Create region.coord by converting python output:
    region.coord = double(vertices);
    
    % Create region.nedges
    region.nedges = zeros(1, region.ne);
    for jj = 1:region.ne
        region.nedges(jj) = length(region.connectivity{jj});
    end
    
    % Extraction of the element coordinates for the kk-th element
    region.coords_element = cell(1, region.ne);
    for kk = 1:region.ne
        region.coords_element{kk} = region.coord(region.connectivity{kk}, :);
    end

    % Create region.id
    region.id = int64(physical_groups);

    % Create region.tag: the mesh is polygonal
    region.tag = 'P';
    
    fprintf('done\n')

    %% Create neighbour and boundary structure

    fprintf('Computing neighbor structure ... ')

    % Create neighbor structure (adjacency)
    [neighbor] = MakeNeighborInternal(region);
    
    % boundary connectivity data and boundary faces tags
    region.connectivity_bc = int64(boundary);
    display(region.connectivity_bc)
    display(isempty(region.connectivity_bc))
    region.bc_tag = int64(b_tags);
    
    % if the boundary could not be gotten from the mesh, compute it:
    if isempty(region.connectivity_bc)
        fprintf('Computing boundary ... ')
        k = 1;
        for i = 1 : region.ne
            for j = 1 : neighbor.nedges(i)
                
                id_edge = neighbor.neigh{i}(j);
                if (id_edge == -1)  % i.e. there is no neighbouring element corresponding to that edge
                    if(j < neighbor.nedges(i))
                        edge = [region.connectivity{i}(j) region.connectivity{i}(j+1)];
                    else
                        edge = [region.connectivity{i}(j) region.connectivity{i}(1)];
                    end
                    region.connectivity_bc(k,1:2) = [edge(1), edge(2)];
                    k = k + 1;
                end
            end
        end

        region.bc_tag=ones(1,length(region.connectivity_bc));
    end
    % change neighbour data to consider the type of boundary conditions
    % (Dirichlet, Neumann, Absorbing):
    [neighbor] = MakeNeighborBoundary(Data,region,neighbor,SimType);

    fprintf('done\n')

    %% Create region.max_kb, region.area and region.BBox
    
    % Clockwise ordering control
    % actually, it sorts them counterclockwise
    % [region] = ClockWiseElements(region);

    region.max_kb = cell(1,length(region.connectivity));

    % Visualization of computational progress
    prog = 0;
    fprintf(1,'Computation Progress of max kb calculation: %3d%%\n',prog);
    
    % Compute element bounding box, area and max_kb 
    for ii = 1:length(region.connectivity)
        
        prog = 100*ii/region.ne;
        fprintf(1,'\b\b\b\b%3.0f%%',prog);

        % Extraction of necessary information
        nedge = length(region.connectivity{ii});
        coords_element = region.coord(region.connectivity{ii},:);
        
        % Computation of bounding box
        region.BBox(ii,1) = min(region.coords_element{ii}(:,1));
        region.BBox(ii,2) = max(region.coords_element{ii}(:,1));
        region.BBox(ii,3) = min(region.coords_element{ii}(:,2));
        region.BBox(ii,4) = max(region.coords_element{ii}(:,2));

        % Computation of element area
        region.area(ii) = polyarea(region.coords_element{ii}(:,1),region.coords_element{ii}(:,2));

        % Preallocation of max_kb structure for each element
        region.max_kb{ii} = zeros(nedge,1);
        % element = polyshape(coords_element(end:-1:1,1), coords_element(end:-1:1,2));

        % Cycle over the jj-th point of the element
        for jj = 1:nedge

            % Extraction of the first point
            v(1,:) = coords_element(jj,:);

            % Extraction of the second point
            if jj == nedge
                v(2,:) = coords_element(1,:);
            else
                v(2,:) = coords_element(jj+1,:);
            end

            % Cycle over all the others points of the element
            for kk = 1:nedge

                if (kk ~= jj) && (kk ~= jj+1)

                    % Extraction of the third point
                    v(3,:) = coords_element(kk,:);

                    % Construction of the triangle and area computation
                    [x_tria, y_tria] = poly2cw( v(:,1), v(:,2));
                    area_tria = polyarea(x_tria, y_tria);

                    % Intersect the triangle and the mesh element
                    % tria = polyshape(x_tria(1:3), y_tria(1:3));
                    % intersection = intersect(element, tria);
                    % x1 = intersection.Vertices(:,1);
                    % y1 = intersection.Vertices(:,2);
                    [x1,y1] = polybool('intersection', coords_element(end:-1:1,1), coords_element(end:-1:1,2), x_tria, y_tria);

                    % Control of the correct construction of the intersection and update of max_kb
                    if (~any(isnan(x1))) && (abs(polyarea(x1,y1) - area_tria) < 1e-6)

                        region.max_kb{ii}(jj) = max(area_tria, region.max_kb{ii}(jj));

                    end
                end
            end
        end
    end
    fprintf('\n')
    
    %% Mesh information saving
    save(output_path,"region","neighbor")
    fprintf('Saved mesh.\n')
    toc
end