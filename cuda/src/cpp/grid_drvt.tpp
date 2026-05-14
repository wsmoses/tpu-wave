// DO ONE THING WELL. 
//     THIS FOLDER IS INTENDED TO WORK for ONE CASE ONLY - FREE SURFACE ON X AND PERIODIC ON Y;
// SPECIALIZE AS MUCH AS NEEDED.


// [2023/06/19]
// NOTE: drvt_to_rths "accumulates" the calculated derivative to the right hand side variable.
//       To simplify the implementation effort, we try to use this same routine for all grids 
//       and all directions, which entails if statements inside. These if statements are made 
//       "if constexpr" for run-time efficiency.
//       
//       v_d is the calculated derivative; i_d is the index associated with the derivative.
//       
//       On Vx and Vy grids, P is (1/rho) on the respective grid. 
//       On Sxy grid, P is mu. 
//       On normal grid, in current implementation, P is lambda; P_extra is 2*mu. The meaning
//       of R_extra depends on the direction, for 'x' direction, R_extra corresponds to Sxx;
//       for 'y' direction, R_extra corresponds to Syy. Although we are accessing (* R_extra)
//       at i_d for an extra time, presumably they are already in the cache (if not register).
//       Therefore, the following implementation may exert more flops, but not (main) memory 
//       access. (In other words, if it's too much changes and too error-prone, it may not be
//       worthwhile to change the two physical parameters to (lambda + 2*mu) and lambda.)
template<bool bool_extra=false>
void drvt_to_rths ( ns_type::host_precision v_d , int i_d , 
                    run_time_vector<ns_type::host_precision> * P ,
                    run_time_vector<ns_type::host_precision> * P_extra ,
                    run_time_vector<ns_type::host_precision> * R_extra )
{
    // add derivative to rhs
    for ( int i_soln = 0; i_soln < this->N_soln; i_soln++ )
        { this->Vec_rths.at(i_soln)(i_d) = this->Vec_rths.at(i_soln)(i_d) + v_d * (* P)(i_d); }
    if constexpr ( bool_extra )
        { (* R_extra)(i_d) = (* R_extra)(i_d) + v_d * (* P_extra)(i_d); }
}
// [2023/07/14]
// NOTE: We should be able to overload operator+= for __half and __nv_bfloat16. We may use two
//       template parameters, one for the variable to be added to (__half or __nv_bfloat16), 
//       the other for the value to be added (can be built-in or library defined). We may need
//       partial template explicit instantiation so that to cause conflict with the operator+=
//       for the built in type. (Or, we can spell out two template functions, one for __half,
//       the other for __nv_bfloat16.)
// [2023/07/17]
// NOTE: I now feel _Float16 from gcc may be the one that has issue with operator+=.

template<bool bool_extra=false>
void calculate_derivative_periodic ( char c_dir )
{
    if ( c_dir != 'y' )
    { 
        printf("%s %d : This function should only be called on y (i.e., 1, i.e., fast) direction.\n", 
                __FILE__, __LINE__); fflush(stdout); exit(0); 
    } // NOTE: This is mostly dictated by the implementation on the GPU side.


    if ( bool_extra == true  ) { assert( N_soln == ns_forward::N_dir && N_prmt == 2 ); }
    if ( bool_extra == false ) { assert( N_soln == 1                 && N_prmt == 1 ); }
    // [2023/06/19]
    // NOTE: If we make the grid class template, we could make N_soln and N_prmt (static) constexpr.


    // pointer (pntr) to the interacting grid on c_dir direction (use 4 letters to align with "this")
    auto & pntr = Map_pntr_grid.at(c_dir);

    // "operators" that "represent" the "matrix"
    run_time_vector<int> & stencil_shift = this->Map_stencil_shift.at(c_dir);
    
    // NOTE: the following S is from the interacting grid
    run_time_vector<ns_type::host_precision> & S = * (this->Map_pntr_soln.at(c_dir) );

    run_time_vector<ns_type::host_precision> * P = & this->Vec_prmt.at(0);
    run_time_vector<ns_type::host_precision> * P_extra = bool_extra ? & this->Vec_prmt.at(1) : nullptr;  // 1 corresponds to 2*mu

    // NOTE: R_extra is from this grid
    run_time_vector<ns_type::host_precision> * R_extra = bool_extra ? & this->Vec_rths.at(1) : nullptr;  // 1 corresponds to 'y'
    // [2023/06/18] NOTE: Dereferencing nullptr may be UB.


    const int & N_modulo = pntr->Map_N_modulo.at(c_dir);    // c_dir is meant to be 'y' here
    // NOTE: I think it should be "pntr" above; but they 
    //       may (should) be the same (i.e., M-grid size 
    //       on that direction).


    constexpr int SL = ns_forward::length_stencil;          // SL : stencil_length


    // WARNING: DO NOT retrieve INT_bound_BGN_x and INT_bound_END_x from the Map
    int INT_bound_BGN_x = 0;  int INT_bound_END_x = this->G_size_x;
    // int INT_bound_BGN_y = 0;  int INT_bound_END_y = this->G_size_y;

    int INT_bound_BGN_y = Map_interior_BGN.at('y');
    int INT_bound_END_y = Map_interior_END.at('y');
    // assuming c_dir is 'y'


    int LFT_bound_BGN_x = 0;  int LFT_bound_END_x = this->G_size_x;
    int LFT_bound_BGN_y = 0;  int LFT_bound_END_y = this->G_size_y;

    
    int RHT_bound_BGN_x = 0;  int RHT_bound_END_x = this->G_size_x;
    int RHT_bound_BGN_y = 0;  int RHT_bound_END_y = this->G_size_y;


    LFT_bound_END_y = INT_bound_BGN_y;  // assuming c_dir is 'y'
    RHT_bound_BGN_y = INT_bound_END_y;  // assuming c_dir is 'y'


    // LFT boundary points
    for ( int ix = LFT_bound_BGN_x; ix < LFT_bound_END_x; ix++ )  // ix goes through the grid index on "this" grid
    for ( int iy = LFT_bound_BGN_y; iy < LFT_bound_END_y; iy++ )  // iy goes through the grid index on "this" grid
    {
        int i_d = ix * this->stride_x + iy * this->stride_y;
        int i_s = ix * pntr->stride_x;

        ns_type::host_precision v_d = 0.;

        // external (->)
        for ( int iw = 0; iw < SL_external - iy; iw++ )
            { v_d = v_d + S( i_s + ( iy + stencil_shift( iw ) + N_modulo ) * pntr->stride_y ) * stencil_dt_dx( iw ); }

        // internal (->)
        for ( int iw = SL_external - iy; iw < SL; iw++ )
            { v_d = v_d + S( i_s + ( iy + stencil_shift( iw ) ) * pntr->stride_y ) * stencil_dt_dx( iw ); }

        drvt_to_rths <bool_extra> ( v_d, i_d, P, P_extra, R_extra );
    }


    // Interior points
    for ( int ix = INT_bound_BGN_x; ix < INT_bound_END_x; ix++ )  // ix goes through the grid index on "this" grid
    for ( int iy = INT_bound_BGN_y; iy < INT_bound_END_y; iy++ )  // iy goes through the grid index on "this" grid
    {
        int i_d = ix * this->stride_x + iy * this->stride_y;
        int i_s = ix * pntr->stride_x;

        ns_type::host_precision v_d = 0.;

        v_d = v_d + S( i_s + ( iy + stencil_shift( 0 ) ) * pntr->stride_y ) * stencil_dt_dx( 0 );
        v_d = v_d + S( i_s + ( iy + stencil_shift( 1 ) ) * pntr->stride_y ) * stencil_dt_dx( 1 );
        v_d = v_d + S( i_s + ( iy + stencil_shift( 2 ) ) * pntr->stride_y ) * stencil_dt_dx( 2 );
        v_d = v_d + S( i_s + ( iy + stencil_shift( 3 ) ) * pntr->stride_y ) * stencil_dt_dx( 3 );

        drvt_to_rths <bool_extra> ( v_d, i_d, P, P_extra, R_extra );
    }


    // RHT boundary points
    for ( int ix = RHT_bound_BGN_x; ix < RHT_bound_END_x; ix++ )  // ix goes through the grid index on "this" grid
    for ( int iy = RHT_bound_BGN_y; iy < RHT_bound_END_y; iy++ )  // iy goes through the grid index on "this" grid
    {
        int i_d = ix * this->stride_x + iy * this->stride_y;
        int i_s = ix * pntr->stride_x;

        ns_type::host_precision v_d = 0.;

        // internal (->)
        for ( int iw = 0; iw < SL_internal + (RHT_bound_END_y - 1) - iy; iw++ )
            { v_d = v_d + S( i_s + ( iy + stencil_shift( iw ) ) * pntr->stride_y ) * stencil_dt_dx( iw ); }

        // external (->)
        for ( int iw = SL_internal + (RHT_bound_END_y - 1) - iy; iw < SL; iw++ )
            { v_d = v_d + S( i_s + ( iy + stencil_shift( iw ) - N_modulo ) * pntr->stride_y ) * stencil_dt_dx( iw ); }

        drvt_to_rths <bool_extra> ( v_d, i_d, P, P_extra, R_extra );
    }

} // calculate_derivative_periodic ()


// NOTE: Above, We have a version of calculate_derivative_periodic () without the 
//       modulo (%) operator working. It almost reduces the run time by a half. 
//       This is intriguing. 
//
//       Is the speed difference more due to the expensiveness of %, or due to
//       memory jump (now being less severe since we separated the for loop into
//       three parts)?
//
//       To test this, we tried to use the same for loop structure to loop through
//       the grid points, but use the % operator within. The speed went back to be
//       the slower end again. This is a strong indication that it is the % operator
//       itself, not because of the for loop structure (and the induced memory 
//       traversal pattern).
//
//       However, it is still unclear whether it is because % itself, or
//       something else it causes. For example, it uses the integer FU, which is
//       also used to calculate address, somehow this can inhibit some efficient
//       memory access mechanism? Or, it could be that because of the appearance
//       of %, SIMD is inhibited.
//
//
         /* I think we need to get help to get to the bottom of this. Maybe it's
         time that we reach out to performance experts. I have one in mind, from
         intel, who is enthusiastic about performance and from the c++ world. */
//
//
//       Clean up is needed.


template<bool bool_extra=false>
void calculate_derivative_periodic_modulo ( char c_dir )
{
    if ( c_dir != 'y' )
    { 
        printf("%s %d : This function should only be called on y (i.e., 1, i.e., fast) direction.\n", 
                __FILE__, __LINE__); fflush(stdout); exit(0); 
    } // NOTE: This is mostly dictated by the implementation on the GPU side.

    if ( bool_extra == true  ) { assert( N_soln == ns_forward::N_dir && N_prmt == 2 ); }
    if ( bool_extra == false ) { assert( N_soln == 1                 && N_prmt == 1 ); }


    // pointer to the interacting grid on c_dir direction (use 4 letters to align with "this")
    auto & pntr = Map_pntr_grid.at(c_dir);

    // "operators" that "represent" the "matrix"
    run_time_vector<int> & stencil_shift = this->Map_stencil_shift.at(c_dir);

    // NOTE: the following S is from the interacting grid
    run_time_vector<ns_type::host_precision> & S = * (this->Map_pntr_soln.at(c_dir) );

    // NOTE: the following R is from this grid
    run_time_vector<ns_type::host_precision> * P = & this->Vec_prmt.at(0);
    run_time_vector<ns_type::host_precision> * P_extra = bool_extra ? & this->Vec_prmt.at(1) : nullptr;  // 1 corresponds to 2*mu

    run_time_vector<ns_type::host_precision> * R_extra = bool_extra ? & this->Vec_rths.at(1) : nullptr;  // 1 corresponds to 'y'
    // [2023/06/18] NOTE: Dereferencing nullptr may be UB.


    // const int stride_dir = pntr->Map_stride.at(c_dir);
    // // NOTE : stride_dir is the change in the 2D vector index when the index on 
    // //        the "dir" direction of the grid is incremented by 1; In the above,
    // //        we retrieved stride_dir from the "pntr" grid, because it is used
    // //        to calculate the 2D vector index on the "pntr" grid. It will be  
    // //        okay to retrieve it from "this" grid as well, since the strides
    // //        AT and AFTER "dir" (in the order of x, y) do not differ between
    // //        these two grids. As its name suggests, stride_dir is the stride 
    // //        AT the "dir" direction.
    // //            Below we have a check to validate this.    
    // if ( this->Map_stride.at(c_dir) != pntr->Map_stride.at(c_dir) ) 
    //     { printf("stride at %c direction should be the same for this and pntr grids.\n", c_dir); fflush(stdout); exit(0); }

    int ix;
    int iy;

    // int & i_dir = c_dir == 'x' ? ix : iy;
    // // NOTE: Reference needs to be defined when declared, thus the ternary operator;
    // //       The alternative is to use pointer:
    // //           int * i_dir; 
    // //           if ( c_dir == 'x' ) { i_dir = & ix; }
    // //           if ( c_dir == 'y' ) { i_dir = & iy; }
    // //       i_dir needs to be dereferenced before using as index.


    int N_modulo = 1<<2;    
    if ( pntr->Map_G_type.at(c_dir) == 'M' ) { N_modulo = pntr->Map_G_size.at(c_dir)    ; }
    if ( pntr->Map_G_type.at(c_dir) == 'N' ) { N_modulo = pntr->Map_G_size.at(c_dir) - 1; }


    // NOTE: For an N grid size of 81 and M grid size of 80, where the first and last grid points 
    //       of the N grid are duplicated, when updating the N grid (using M grid values), the above
    //       modulo operation maps the indices in the following way:
    // 
    //              -2  -1  |   0                           79    |  80  81  
    //                      | 0   1                      79    80 | 
    //              78  79                                            0   1
    //
    //       when updating the M grid (using N grid values), the above modulo operation maps the indices
    //       in the following way:
    //
    //                      |   0                           79    |    
    //                  -1  | 0   1                      79    80 |  81
    //                  79    0                                 0     1
    //
    //       Notice that 80, although within range, gets mapped to 0, i.e., the last grid point gets mapped
    //       to the first grid point when used to calculate derviatives and update other fields.
    //
    //       There is some subtlety on this when source lands on the duplicated boundary in calculating the
    //       energy. We should keep this mapping and "forbid" the source to be placed on the last grid point
    //       in the input file. This way, we do not need to test if the source is on duplicated boundary and
    //       make corresponding changes.
    //
    //           See also grid_energy.cpp (adjust_parameters_energy_periodic) and forward_simulation.cpp 
    //       (forward_simulation_periodic_horizontal) for related comments.


    // Interior points

    // ---- loop bounds
    const int INT_bound_BGN_x = 0;  const int INT_bound_END_x = this->G_size_x;
    const int INT_bound_BGN_y = 0;  const int INT_bound_END_y = this->G_size_y;

    // ---- calculation
    for ( ix = INT_bound_BGN_x; ix < INT_bound_END_x; ix++ )  // ix goes through the grid index on "this" grid
    for ( iy = INT_bound_BGN_y; iy < INT_bound_END_y; iy++ )  // iy goes through the grid index on "this" grid
    {
        int i_d = ix * this->stride_x + iy * this->stride_y;
        int i_s = ix * pntr->stride_x;

        ns_type::host_precision v_d = 0.;

        v_d = v_d + S( i_s + ( ( iy + stencil_shift( 0 ) + N_modulo ) % N_modulo ) * pntr->stride_y ) * stencil_dt_dx( 0 );
        v_d = v_d + S( i_s + ( ( iy + stencil_shift( 1 ) + N_modulo ) % N_modulo ) * pntr->stride_y ) * stencil_dt_dx( 1 );
        v_d = v_d + S( i_s + ( ( iy + stencil_shift( 2 ) + N_modulo ) % N_modulo ) * pntr->stride_y ) * stencil_dt_dx( 2 );
        v_d = v_d + S( i_s + ( ( iy + stencil_shift( 3 ) + N_modulo ) % N_modulo ) * pntr->stride_y ) * stencil_dt_dx( 3 );

        drvt_to_rths <bool_extra> ( v_d, i_d, P, P_extra, R_extra );
    }
    
} // calculate_derivative_periodic_modulo ()


template<bool bool_extra=false>
void calculate_derivative ( char c_dir )
{
    if ( c_dir != 'x' )
        { printf("This function should only be called on x (i.e., 0, i.e., slow) direction.\n"); fflush(stdout); exit(0); }
    // NOTE: This is mostly dictated by the implementation on the GPU side.

    if ( bool_extra == true  ) { assert( N_soln == ns_forward::N_dir && N_prmt == 2 ); }
    if ( bool_extra == false ) { assert( N_soln == 1                 && N_prmt == 1 ); }


    // pointer to the interacting grid on c_dir direction (use 4 letters to align with "this")
    auto & pntr = Map_pntr_grid.at(c_dir);

    // "operators" that "represent" the "matrix"
    run_time_vector<int> & stencil_shift = this->Map_stencil_shift.at(c_dir);
    
    run_time_matrix<ns_type::host_precision> & D_bdry_L = this->Map_D_bdry.at( {c_dir , 'L'} );
    run_time_matrix<ns_type::host_precision> & D_bdry_R = this->Map_D_bdry.at( {c_dir , 'R'} );

    // NOTE: the following S is from the interacting grid
    run_time_vector<ns_type::host_precision> & S = * (this->Map_pntr_soln.at(c_dir) );
    
    // NOTE: the following R is from this grid
    run_time_vector<ns_type::host_precision> * P = & this->Vec_prmt.at(0);
    run_time_vector<ns_type::host_precision> * P_extra = bool_extra ? & this->Vec_prmt.at(1) : nullptr;  // 1 corresponds to 2*mu

    run_time_vector<ns_type::host_precision> * R_extra = bool_extra ? & this->Vec_rths.at(0) : nullptr;  // 0 corresponds to 'x'
    // [2023/06/18] NOTE: Dereferencing nullptr may be UB.


    const int stride_dir = pntr->Map_stride.at(c_dir);
    // NOTE : stride_dir is the change in the 2D vector index when the index on 
    //        the "dir" direction of the grid is incremented by 1; In the above,
    //        we retrieved stride_dir from the "pntr" grid, because it is used
    //        to calculate the 2D vector index on the "pntr" grid. It will be  
    //        okay to retrieve it from "this" grid as well, since the strides
    //        AT and AFTER "dir" (in the order of x, y) do not differ between
    //        these two grids. As its name suggests, stride_dir is the stride 
    //        AT the "dir" direction.
    //            Below we have a check to validate this.    
    if ( this->Map_stride.at(c_dir) != pntr->Map_stride.at(c_dir) ) 
        { printf("stride at %c direction should be the same for this and pntr grids.\n", c_dir); fflush(stdout); exit(0); }

    int ix;
    int iy;

    int & i_dir = c_dir == 'x' ? ix : iy;
    // NOTE: Reference needs to be defined when declared, thus the ternary operator;
    //       The alternative is to use pointer:
    //           int * i_dir; 
    //           if ( c_dir == 'x' ) { i_dir = & ix; }
    //           if ( c_dir == 'y' ) { i_dir = & iy; }
    //       i_dir needs to be dereferenced before using as index.


    // Left bdry points

    // ---- loop bounds
    const int LFT_bound_BGN_x = 0;  const int LFT_bound_END_x = c_dir == 'x' ? D_bdry_L.rows : this->G_size_x;
    const int LFT_bound_BGN_y = 0;  const int LFT_bound_END_y = c_dir == 'y' ? D_bdry_L.rows : this->G_size_y;
    // NOTE: Using ternary operator is less expressive than if statements, but may make things easier if
    //       we want to make a constexpr version (just in case).

    const int LFT_bound_BGN_w = 0;    const int LFT_bound_END_w = D_bdry_L.cols;

    // ---- calculation
    for ( ix = LFT_bound_BGN_x; ix < LFT_bound_END_x; ix++ )  // ix goes through the grid index on "this" grid
    for ( iy = LFT_bound_BGN_y; iy < LFT_bound_END_y; iy++ )  // iy goes through the grid index on "this" grid
    {
        int i_d = ix * this->stride_x + iy * this->stride_y;
        ns_type::host_precision v_d = 0.;

        for ( int iw = LFT_bound_BGN_w; iw < LFT_bound_END_w; iw++ )  // iw goes through the grid index on "pntr" grid
        {
            int i_s = ix * pntr->stride_x + iy * pntr->stride_y + ( iw - i_dir ) * stride_dir;
            // NOTE: i_s the the 'col' index of the "matrix", which is the input index in foward, and corresponds to 
            //       "pntr" grid in forward; 
            //           On the other hand, ix and iy go through the grid indices on "this" grid; to calculate i_s, 
            //       we need to switch the grid index on "dir" direction from i_dir to iw (which goes through the 'col' 
            //       index of the "matrix" and corresponds to "pntr" grid as i_s); indices for the other two directions 
            //       are unchanged between "this" and "pntr" and need no modification. 
            //           The trick is to add the term ( iw - i_dir ) * stride_dir, which effectively does the switch
            //       and avoids if statements.

            v_d = v_d + S( i_s ) * D_bdry_L( i_dir , iw );
        }

        drvt_to_rths <bool_extra> ( v_d, i_d, P, P_extra, R_extra );
    }
    

    // Interior points

    // ---- loop bounds
    const int INT_bound_BGN_x = c_dir == 'x' ? D_bdry_L.rows : 0;  const int INT_bound_END_x = c_dir == 'x' ? this->G_size_x - D_bdry_R.rows : this->G_size_x;
    const int INT_bound_BGN_y = c_dir == 'y' ? D_bdry_L.rows : 0;  const int INT_bound_END_y = c_dir == 'y' ? this->G_size_y - D_bdry_R.rows : this->G_size_y;

    // ---- calculation
    for ( ix = INT_bound_BGN_x; ix < INT_bound_END_x; ix++ )  // ix goes through the grid index on "this" grid
    for ( iy = INT_bound_BGN_y; iy < INT_bound_END_y; iy++ )  // iy goes through the grid index on "this" grid
    {
        int i_d = ix * this->stride_x + iy * this->stride_y;  // 'row' index of the "matrix" <-> output index <-> "this" grid in forward
        int i_s = ix * pntr->stride_x + iy * pntr->stride_y;  // 'col' index of the "matrix" <->  input index <-> "pntr" grid in forward

        ns_type::host_precision v_d = 0.;

        v_d = v_d + S( i_s + stencil_shift( 0 ) * stride_dir ) * stencil_dt_dx( 0 );
        v_d = v_d + S( i_s + stencil_shift( 1 ) * stride_dir ) * stencil_dt_dx( 1 );
        v_d = v_d + S( i_s + stencil_shift( 2 ) * stride_dir ) * stencil_dt_dx( 2 );
        v_d = v_d + S( i_s + stencil_shift( 3 ) * stride_dir ) * stencil_dt_dx( 3 );

        drvt_to_rths <bool_extra> ( v_d, i_d, P, P_extra, R_extra );        
    }


    // Right bdry points

    // ---- loop bounds
    const int RHT_bound_BGN_x = c_dir == 'x' ? this->G_size_x - D_bdry_R.rows : 0;  const int RHT_bound_END_x = this->G_size_x;
    const int RHT_bound_BGN_y = c_dir == 'y' ? this->G_size_y - D_bdry_R.rows : 0;  const int RHT_bound_END_y = this->G_size_y;
    // NOTE: Using ternary operator is less expressive than if statements, but may make things easier if
    //       we want to make a constexpr version (just in case).

    const int RHT_bound_BGN_w   = pntr->Map_G_size.at(c_dir) - D_bdry_R.cols;  const int RHT_bound_END_w = pntr->Map_G_size.at(c_dir);

    const int RHT_bound_BGN_dir = this->Map_G_size.at(c_dir) - D_bdry_R.rows;
    
    // ---- calculation
    for ( ix = RHT_bound_BGN_x; ix < RHT_bound_END_x; ix++ )  // ix goes through the grid index on "this" grid
    for ( iy = RHT_bound_BGN_y; iy < RHT_bound_END_y; iy++ )  // iy goes through the grid index on "this" grid
    {
        int i_d = ix * this->stride_x + iy * this->stride_y;
        ns_type::host_precision v_d = 0.;

        for ( int iw = RHT_bound_BGN_w; iw < RHT_bound_END_w; iw++ )  // iw goes through the grid index on "pntr" grid
        {

            int i_s = ix * pntr->stride_x + iy * pntr->stride_y + ( iw - i_dir ) * stride_dir;
            // NOTE: i_s the the 'col' index of the "matrix", which is the input index in foward, and corresponds to 
            //       "pntr" grid in forward; 
            //           On the other hand, ix and iy go through the grid indices on "this" grid; to calculate i_s, 
            //       we need to switch the grid index on "dir" direction from i_dir to iw (which goes through the 'col' 
            //       index of the "matrix" and corresponds to "pntr" grid as i_s); indices for the other two directions 
            //       are unchanged between "this" and "pntr" and need no modification. 
            //           The trick is to add the term ( iw - i_dir ) * stride_dir, which effectively does the switch
            //       and avoids if statements.

            v_d = v_d + S( i_s ) * D_bdry_R( i_dir - RHT_bound_BGN_dir , iw - RHT_bound_BGN_w );
        }

        drvt_to_rths <bool_extra> ( v_d, i_d, P, P_extra, R_extra );
    }

} // calculate_derivative ()


template<bool bool_extra=false>
void secure_bdry ( char c_dir )
{
    if ( c_dir != 'x' )
        { printf("This function should only be called on x (i.e., 0, i.e., slow) direction.\n"); fflush(stdout); exit(0); }
    // NOTE: This is mostly dictated by the implementation on the GPU side.

    if ( bool_extra == true  ) { assert( N_soln == ns_forward::N_dir && N_prmt == 2 ); }
    if ( bool_extra == false ) { assert( N_soln == 1                 && N_prmt == 1 ); }


    // NOTE: the following S is from the interacting grid
    run_time_vector<ns_type::host_precision> & S = * (this->Map_pntr_soln.at(c_dir) );

    // NOTE: the following R is from this grid
    run_time_vector<ns_type::host_precision> * P = & this->Vec_prmt.at(0);
    run_time_vector<ns_type::host_precision> * P_extra = bool_extra ? & this->Vec_prmt.at(1) : nullptr;  // 1 corresponds to 2*mu
    
    run_time_vector<ns_type::host_precision> * R_extra = bool_extra ? & this->Vec_rths.at(0) : nullptr;  // 0 corresponds to 'x'
    // [2023/06/18] NOTE: Dereferencing nullptr may be UB.


    // pointer to the interacting grid on c_dir direction (use 4 letters to align with "this")
    auto & pntr = Map_pntr_grid.at(c_dir);


    // NOTE: Map_projection has been prepared so that projection_* operators are 
    //       compatible to be applied on S (which is from the interacting grid).
    run_time_vector<ns_type::host_precision> & projection_L = this->Map_projection.at({c_dir,'L'});
    run_time_vector<ns_type::host_precision> & projection_R = this->Map_projection.at({c_dir,'R'});

    run_time_vector<ns_type::host_precision> & A_inv_projection_L = this->Map_A_inv_projection.at({c_dir,'L'});
    run_time_vector<ns_type::host_precision> & A_inv_projection_R = this->Map_A_inv_projection.at({c_dir,'R'});


    // ---- Memory spaces used to store the 'projected' solutions (on this MPI region)
    run_time_vector<ns_type::host_precision> & SEND_L = this->Map_buffer_SEND.at(c_dir).at('L');  SEND_L.set_constant(0.);
    run_time_vector<ns_type::host_precision> & SEND_R = this->Map_buffer_SEND.at(c_dir).at('R');  SEND_R.set_constant(0.);
    // ---- Memory spaces used to store the 'projected' solutions (from the opposing MPI region)
    run_time_vector<ns_type::host_precision> & RECV_L = this->Map_buffer_RECV.at(c_dir).at('L');  RECV_L.set_constant(0.);
    run_time_vector<ns_type::host_precision> & RECV_R = this->Map_buffer_RECV.at(c_dir).at('R');  RECV_R.set_constant(0.);
    //
    // NOTE: Since no MPI is used in forward simulation, the above 'duplicated' memory spaces 
    //       and the copy operations later are not necessary. They are kept for API stability.
    

    const int stride_dir = Map_stride.at(c_dir);
    // NOTE : stride_dir is the change in the 2D vector index when the index on 
    //        the "dir" direction of the grid is incremented by 1; In the above,
    //        we retrieved stride_dir from the "pntr" grid, because it is used
    //        to calculate the 2D vector index on the "pntr" grid. It will be  
    //        okay to retrieve it from "this" grid as well, since the strides
    //        AT and AFTER "dir" (in the order of x, y) do not differ between
    //        these two grids. As its name suggests, stride_dir is the stride 
    //        AT the "dir" direction.
    //            Below we have a check to validate this.    
    if ( this->Map_stride.at(c_dir) != pntr->Map_stride.at(c_dir) ) 
        { printf("stride at %c direction should be the same for this and pntr grids.\n", c_dir); fflush(stdout); exit(0); }

    int ix;
    int iy;

    int & i_dir = c_dir == 'x' ? ix : iy;
    // NOTE: Reference needs to be defined when declared, thus the ternary operator;
    //       The alternative is to use pointer:
    //           int * i_dir; 
    //           if ( c_dir == 'x' ) { i_dir = & ix; }
    //           if ( c_dir == 'y' ) { i_dir = & iy; }
    //       i_dir needs to be dereferenced before using as index.


    // Project data to the left boundary 

    // ---- loop bounds
    const int LFT_bound_BGN_x = 0;  const int LFT_bound_END_x = c_dir == 'x' ? 1 : pntr->G_size_x;
    const int LFT_bound_BGN_y = 0;  const int LFT_bound_END_y = c_dir == 'y' ? 1 : pntr->G_size_y;
    // NOTE: Using ternary operator is less expressive than if statements, but may make things 
    //       easier if we want to make a constexpr version (just in case).

    int LFT_bound_BGN_w = 0;
    int LFT_bound_END_w = projection_L.length;

    // ---- projection
    if ( params.Map_bdry_type.at({ c_dir , 'L' }) != 'F'                                  // if the boundary is 'internal' (MPI boundary)
    || ( params.Map_bdry_type.at({ c_dir , 'L' }) == 'F' && this->free_surface_update ) ) // if the boundary is 'external' and THIS grid
    {                                                                                     // requires penalty terms to update (Vx,Vy)
        int i_p = 0;

        for ( ix = LFT_bound_BGN_x; ix < LFT_bound_END_x; ix++ )  // ix goes through the grid index for 'a' plane perpendicular to "dir" direction
        for ( iy = LFT_bound_BGN_y; iy < LFT_bound_END_y; iy++ )  // iy goes through the grid index for 'a' plane perpendicular to "dir" direction
        {
            for ( int iw = LFT_bound_BGN_w; iw < LFT_bound_END_w; iw++ )  // iw goes through the nonzero entries of the "projection" operator;
            {                                                             // this means it goes through the grid index on "pntr" grid since the 
                                                                          // "projection" operator is applied on S, which is from "pntr" grid        
                int i_s = ix * pntr->stride_x + iy * pntr->stride_y + ( iw - i_dir ) * stride_dir;
                // NOTE: For the extra term ( iw - i_dir ) * stride_dir, see 
                //       comments in calculate_derivative () and above.

                SEND_L( i_p ) = SEND_L( i_p ) + projection_L( iw ) * S( i_s );
            }
            i_p++;
        }
    }


    // project data to the right boundary

    // ---- loop bounds
    const int RHT_bound_BGN_x = c_dir == 'x' ? pntr->G_size_x - 1 : 0;  const int RHT_bound_END_x = pntr->G_size_x;
    const int RHT_bound_BGN_y = c_dir == 'y' ? pntr->G_size_y - 1 : 0;  const int RHT_bound_END_y = pntr->G_size_y;
    // NOTE: Using ternary operator is less expressive than if statements, but may make things 
    //       easier if we want to make a constexpr version (just in case).

    int RHT_bound_BGN_w = pntr->Map_G_size.at(c_dir) - projection_R.length;
    int RHT_bound_END_w = pntr->Map_G_size.at(c_dir);

    // ---- projection
    if ( params.Map_bdry_type.at({ c_dir , 'R' }) != 'F'                                  // if the boundary is 'internal' (MPI boundary)
    || ( params.Map_bdry_type.at({ c_dir , 'R' }) == 'F' && this->free_surface_update ) ) // if the boundary is 'external' and THIS grid
    {                                                                                     // requires penalty terms to update (Vx,Vy)
        int i_p = 0;

        for ( ix = RHT_bound_BGN_x; ix < RHT_bound_END_x; ix++ )
        for ( iy = RHT_bound_BGN_y; iy < RHT_bound_END_y; iy++ )
        {
            for ( int iw = RHT_bound_BGN_w; iw < RHT_bound_END_w; iw++ )
            {
                int i_s = ix * pntr->stride_x + iy * pntr->stride_y + ( iw - i_dir ) * stride_dir;
                // NOTE: For the extra term ( iw - i_dir ) * stride_dir, see 
                //       comments in calculate_derivative () and above.

                SEND_R( i_p ) = SEND_R( i_p ) + projection_R( iw - RHT_bound_BGN_w ) * S( i_s );
            }
            i_p++;
        }
    }


    // copy (send) data for the left boundary update and set the penalty parameter
    ns_type::host_precision eta_L = 0.; 
    if ( params.Map_bdry_type.at({ c_dir , 'L' }) != 'F'                              ) { eta_L =   1./2.; RECV_L = SEND_R;         }
    if ( params.Map_bdry_type.at({ c_dir , 'L' }) == 'F' && this->free_surface_update ) { eta_L =   1.   ; RECV_L.set_constant(0.); }
    
    // copy (send) data for the right boundary update and set the penalty parameter
    ns_type::host_precision eta_R = 0.;
    if ( params.Map_bdry_type.at({ c_dir , 'R' }) != 'F'                              ) { eta_R = - 1./2.; RECV_R = SEND_L;         }
    if ( params.Map_bdry_type.at({ c_dir , 'R' }) == 'F' && this->free_surface_update ) { eta_R = - 1.   ; RECV_R.set_constant(0.); }


    // update the left boundary

    LFT_bound_BGN_w = 0;
    LFT_bound_END_w = A_inv_projection_L.length;

    if ( params.Map_bdry_type.at({ c_dir , 'L' }) != 'F'                                  // if the boundary is 'internal' (MPI boundary)
    || ( params.Map_bdry_type.at({ c_dir , 'L' }) == 'F' && this->free_surface_update ) ) // if the boundary is 'external' and THIS grid
    {                                                                                        // requires penalty terms to update (Vx,Vy)
        int i_p = 0;

        for ( ix = LFT_bound_BGN_x; ix < LFT_bound_END_x; ix++ )
        for ( iy = LFT_bound_BGN_y; iy < LFT_bound_END_y; iy++ )
        {
            for ( int iw = LFT_bound_BGN_w; iw < LFT_bound_END_w; iw++ )
            {
                int i_d = ix * this->stride_x + iy * this->stride_y + ( iw - i_dir ) * stride_dir;
                // NOTE: For the extra term ( iw - i_dir ) * stride_dir, see 
                //       comments in calculate_derivative () and above.

                ns_type::host_precision v_d = eta_L * A_inv_projection_L( iw ) * ( SEND_L( i_p ) - RECV_L( i_p ) );
                drvt_to_rths <bool_extra> ( v_d, i_d, P, P_extra, R_extra );
            }
            i_p++;            
        }
    }


    // update the right boundary

    RHT_bound_BGN_w = this->Map_G_size.at(c_dir) - A_inv_projection_R.length;
    RHT_bound_END_w = this->Map_G_size.at(c_dir);

    if ( params.Map_bdry_type.at({ c_dir , 'R' }) != 'F'                                  // if the boundary is 'internal' (MPI boundary)
    || ( params.Map_bdry_type.at({ c_dir , 'R' }) == 'F' && this->free_surface_update ) ) // if the boundary is 'external' and THIS grid
    {                                                                                        // requires penalty terms to update (Vx,Vy)
        int i_p = 0;

        for ( ix = RHT_bound_BGN_x; ix < RHT_bound_END_x; ix++ )
        for ( iy = RHT_bound_BGN_y; iy < RHT_bound_END_y; iy++ )
        {
            for ( int iw = RHT_bound_BGN_w; iw < RHT_bound_END_w; iw++ )
            {
                int i_d = ix * this->stride_x + iy * this->stride_y + ( iw - i_dir ) * stride_dir;
                // NOTE: For the extra term ( iw - i_dir ) * stride_dir, see 
                //       comments in calculate_derivative () and above.

                ns_type::host_precision v_d = eta_R * A_inv_projection_R( iw - RHT_bound_BGN_w ) * ( SEND_R( i_p ) - RECV_R( i_p ) );

                drvt_to_rths <bool_extra> ( v_d, i_d, P, P_extra, R_extra );
            }
            i_p++;
        }        
    }

} // secure_bdry ()