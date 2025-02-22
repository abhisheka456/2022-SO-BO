#include "de.h"

Fitness S_LSHADE_DP::run(double *fitness_memory)
{
  cout << scientific << setprecision(6);
  initializeParameters();
  setSHADEParameters();

  int fit_mem_index = 0;
  int FE_term = -1;
  count_fes = 0;

  // cout << pop_size << endl;
  // cout << arc_size << endl;
  // cout << p_best_rate << endl;
  // cout << memory_size << endl;

  vector<Individual> pop;
  vector<Fitness> fitness(pop_size, 0);
  vector<Individual> children;
  vector<Fitness> children_fitness(pop_size, 0);
  vector<int> stagnation(pop_size, 0);

  //initialize population
  for (int i = 0; i < pop_size; i++)
  {
    pop.push_back(makeNewIndividual());
    children.push_back((variable *)malloc(sizeof(variable) * problem_size));
  }

  // evaluate the initial population's fitness values
  evaluatePopulation(pop, fitness);

  Individual bsf_solution = (variable *)malloc(sizeof(variable) * problem_size);
  Fitness bsf_fitness;

  if ((fitness[0] - optimum) <= epsilon)
  {
    fitness[0] = optimum + epsilon;
    FE_term = 1;
  }

  bsf_fitness = fitness[0];
  for (int j = 0; j < problem_size; j++)
    bsf_solution[j] = pop[0][j];
  /////////////////////////////////////////////////////////////////////////
  for (int i = 0; i < pop_size; i++)
  {
    if ((fitness[i] - optimum) <= epsilon)
    {
      fitness[i] = optimum + epsilon;
      FE_term = i;
    }

    if (fitness[i] < bsf_fitness)
    {
      bsf_fitness = fitness[i];
      for (int j = 0; j < problem_size; j++)
        bsf_solution[j] = pop[i][j];
    }
  }
  ////////////////////////////////////////////////////////////////////////////

  //for external archive
  int arc_ind_count = 0;
  int random_selected_arc_ind;
  vector<Individual> archive;
  for (int i = 0; i < arc_size; i++)
    archive.push_back((variable *)malloc(sizeof(variable) * problem_size));

  vector<int> num_success_params(M);
  vector<vector<variable>> success_sf(M);
  vector<vector<variable>> dif_fitness(M);

  vector<vector<variable>> memory_sf;
  for (int i = 0; i < M; i++)
  {
    vector<variable> mem_f_i(memory_size, 0.5);
    memory_sf.push_back(mem_f_i);
  }

  variable temp_sum_sf;
  variable sum;
  variable weight;

  vector<variable> improve_fitness(M, 0.0);
  vector<int> consumed_fes(M, 0);
  int best_op = 0; //rand() % M;

  //memory index counter
  vector<int> memory_pos(M, 0);

  //for new parameters sampling
  variable mu_sf;
  int random_selected_period;
  int *mut_op = (int *)malloc(sizeof(int) * pop_size);
  variable *pop_sf = (variable *)malloc(sizeof(variable) * pop_size);
  variable *pop_cr = (variable *)malloc(sizeof(variable) * pop_size);

  //for current-to-pbest/1
  int p_best_ind;
  int p_num = round(pop_size * p_best_rate);
  int *sorted_array = (int *)malloc(sizeof(int) * pop_size);
  Fitness *temp_fit = (Fitness *)malloc(sizeof(Fitness) * pop_size);

  // for linear population size reduction
  int max_pop_size = pop_size;
  int min_pop_size = 4;
  int plan_pop_size;

  //main loop
  int generation = 0;
  while (count_fes < max_num_evaluations && FE_term == -1)
  {
    generation++;

    for (int i = 0; i < pop_size; i++)
      sorted_array[i] = i;
    for (int i = 0; i < pop_size; i++)
      temp_fit[i] = fitness[i];
    sortIndexWithQuickSort(&temp_fit[0], 0, pop_size - 1, sorted_array);

    for (int target = 0; target < pop_size; target++)
    {
      // select a mutation operator
      int op_code = -1;
      variable rand_dbl = randDouble();
      for (int op = 0; op < M; op++)
      {
        if (op * gamma <= rand_dbl && rand_dbl < (op + 1) * gamma)
        {
          op_code = op;
          break;
        }
      }

      if (op_code == -1)
        op_code = best_op;

      mut_op[target] = op_code;
      consumed_fes[op_code]++;

      //In each generation, CR_i and F_i used by each individual x_i are generated by first selecting an index r_i randomly from [1, H]
      random_selected_period = rand() % memory_size;
      mu_sf = memory_sf[op_code][random_selected_period];

      //generate F_i and repair its value
      do
      {
        pop_sf[target] = cauchy_g(mu_sf, 0.1);
      } while (pop_sf[target] <= 0);

      if (pop_sf[target] > 1)
        pop_sf[target] = 1;

      // generate CR_i
      if (count_fes <= 0.5 * max_num_evaluations || op_code == 0)
      {
        pop_cr[target] = 0;
      }
      else
      {
        pop_cr[target] = randDouble();
      }

      if (op_code == 0)
      {
        operateTarget1BinWithArchive(pop, &children[target][0], target, pop_sf[target], pop_cr[target], archive, arc_ind_count);
      }
      else if (op_code == 1)
      {
        //p-best individual is randomly selected from the top pop_size *  p_i members
        p_best_ind = sorted_array[rand() % p_num];
        operateCurrentToPBest1BinWithArchive(pop, &children[target][0], target, p_best_ind, pop_sf[target], pop_cr[target], archive, arc_ind_count);
      }
    }

    // evaluate the children's fitness values
    evaluatePopulation(children, children_fitness);

    /////////////////////////////////////////////////////////////////////////
    //update the bsf-solution and check the current number of fitness evaluations
    // if the current number of fitness evaluations over the max number of fitness evaluations, the search is terminated
    // So, this program is unconcerned about L-SHADE algorithm directly
    for (int i = 0; i < pop_size; i++)
    {
      if ((children_fitness[i] - optimum) <= epsilon)
      {
        children_fitness[i] = optimum + epsilon;
        FE_term = count_fes;
      }

      if (children_fitness[i] < bsf_fitness)
      {
        bsf_fitness = children_fitness[i];
        for (int j = 0; j < problem_size; j++)
          bsf_solution[j] = children[i][j];
      }
    }
    ////////////////////////////////////////////////////////////////////////////

    //generation alternation
    for (int i = 0; i < pop_size; i++)
    {
      if (children_fitness[i] == fitness[i])
      {
        fitness[i] = children_fitness[i];
        for (int j = 0; j < problem_size; j++)
        {
          pop[i][j] = children[i][j];
        }

        if (count_fes >= 0.5 * max_num_evaluations)
        {
          int count = 0;
          for (int j = 0; j < problem_size; j++)
          {
            if (children[i][j] != pop[i][j])
            {
              count++;
            }
          }

          if (count > 0)
          {
            stagnation[i] = 0;
          }
          else
          {
            stagnation[i]++;
          }
        }
      }
      else if (children_fitness[i] < fitness[i])
      {
        //parent vectors x_i which were worse than the trial vectors u_i are preserved
        if (arc_size > 1)
        {
          if (arc_ind_count < arc_size)
          {
            for (int j = 0; j < problem_size; j++)
              archive[arc_ind_count][j] = pop[i][j];
            arc_ind_count++;
          }
          //Whenever the size of the archive exceeds, randomly selected elements are deleted to make space for the newly inserted elements
          else
          {
            random_selected_arc_ind = rand() % arc_size;
            for (int j = 0; j < problem_size; j++)
              archive[random_selected_arc_ind][j] = pop[i][j];
          }
        }

        dif_fitness[mut_op[i]].push_back(fabs(fitness[i] - children_fitness[i]));
        improve_fitness[mut_op[i]] += fabs(fitness[i] - children_fitness[i]);

        fitness[i] = children_fitness[i];
        for (int j = 0; j < problem_size; j++)
          pop[i][j] = children[i][j];

        //successful parameters are preserved in S_F and S_CR
        success_sf[mut_op[i]].push_back(pop_sf[i]);

        stagnation[i] = 0;
      }
      else
      {
        if (count_fes >= 0.5 * max_num_evaluations)
        {
          stagnation[i]++;
        }
      }
    }

    for (int i = 0; i < M; i++)
      num_success_params[i] = success_sf[i].size();

    // if numeber of successful parameters > 0, historical memories are updated
    for (int m = 0; m < M; m++)
    {
      if (num_success_params[m] > 0)
      {
        memory_sf[m][memory_pos[m]] = 0;
        temp_sum_sf = 0;
        sum = 0;

        for (int i = 0; i < num_success_params[m]; i++)
          sum += dif_fitness[m][i];

        //weighted lehmer mean
        for (int i = 0; i < num_success_params[m]; i++)
        {
          weight = dif_fitness[m][i] / sum;

          memory_sf[m][memory_pos[m]] += weight * success_sf[m][i] * success_sf[m][i];
          temp_sum_sf += weight * success_sf[m][i];
        }

        memory_sf[m][memory_pos[m]] /= temp_sum_sf;

        //increment the counter
        memory_pos[m]++;
        if (memory_pos[m] >= memory_size)
          memory_pos[m] = 0;

        //clear out the S_F, S_CR and delta fitness
        success_sf[m].clear();
        dif_fitness[m].clear();
      }
    }

    // update best mutation operator
    if (generation % NG == 0)
    {
      // cout << generation << " " << best_op << " " << consumed_fes[0] << "  " << consumed_fes[1] << endl;
      int new_best_op = -1;
      double best_improve_rate = 0;

      for (int i = 0; i < M; i++)
      {
        if (consumed_fes[i] > 0)
        {
          double improve_rate = improve_fitness[i] / consumed_fes[i];

          if (improve_rate > best_improve_rate)
          {
            best_improve_rate = improve_rate;
            new_best_op = i;
          }
        }

        consumed_fes[i] = 0;
        improve_fitness[i] = 0;
      }

      if (new_best_op == -1)
      {
        best_op = 0; //rand() % M;
      }
      else
      {
        best_op = new_best_op;
      }
    }

    // calculate the population size in the next generation
    plan_pop_size = round((((min_pop_size - max_pop_size) / (double)max_num_evaluations) * count_fes) + max_pop_size);

    if (pop_size > plan_pop_size)
    {
      reduction_ind_num = pop_size - plan_pop_size;
      if (pop_size - reduction_ind_num < min_pop_size)
        reduction_ind_num = pop_size - min_pop_size;

      reducePopulationWithSort(pop, fitness, stagnation);

      // resize the archive size
      arc_size = pop_size * g_arc_rate;
      if (arc_ind_count > arc_size)
        arc_ind_count = arc_size;

      // resize the number of p-best individuals
      // p_best_rate = 0.4 + (0.2 - 0.4) * count_fes * 1.0 / max_num_evaluations;
      p_num = round(pop_size * p_best_rate);
      if (p_num <= 1)
        p_num = 2;
    }

    // dynamic perturbation
    for (int i = 0; i < pop_size; i++)
    {
      if (stagnation[i] > 100)
      {
        double alpha = count_fes * 1.0 / max_num_evaluations;

        for (int j = 0; j < problem_size; j++)
        {
          if (randDouble() <= 0.5)
          {
            double rand_x = min_region + randDouble() * (max_region - min_region);
            pop[i][j] = alpha * pop[i][j] + (1.0 - alpha) * rand_x;
          }
        }

        evaluateIndividual(pop[i], &fitness[i]);
        stagnation[i] = 0;
      }
    }

    if (count_fes >= g_max_num_evaluations * pow(g_problem_size, fit_mem_index / 5.0 - 3.0))
    {
      // cout << bsf_fitness << endl;
      fitness_memory[fit_mem_index++] = bsf_fitness - optimum;
    }
  }

  if (FE_term == -1)
  {
    FE_term = max_num_evaluations;
  }
  else
  {
    for (; fit_mem_index < 16; fit_mem_index++)
    {
      fitness_memory[fit_mem_index] = epsilon;
    }
  }
  fitness_memory[fit_mem_index] = FE_term;

  return bsf_fitness - optimum;
}

void S_LSHADE_DP::operateCurrentToPBest1BinWithArchive(const vector<Individual> &pop, Individual child, int &target, int &p_best_individual, variable &scaling_factor, variable &cross_rate, const vector<Individual> &archive, int &arc_ind_count)
{
  int r1, r2;

  do
  {
    r1 = rand() % pop_size;
  } while (r1 == target);
  do
  {
    r2 = rand() % (pop_size + arc_ind_count);
  } while ((r2 == target) || (r2 == r1));

  int random_variable = rand() % problem_size;

  if (r2 >= pop_size)
  {
    r2 -= pop_size;
    for (int i = 0; i < problem_size; i++)
    {
      if ((randDouble() < cross_rate) || (i == random_variable))
      {
        child[i] = pop[target][i] + scaling_factor * (pop[p_best_individual][i] - pop[target][i]) + scaling_factor * (pop[r1][i] - archive[r2][i]);
      }
      else
      {
        child[i] = pop[target][i];
      }
    }
  }
  else
  {
    for (int i = 0; i < problem_size; i++)
    {
      if ((randDouble() < cross_rate) || (i == random_variable))
      {
        child[i] = pop[target][i] + scaling_factor * (pop[p_best_individual][i] - pop[target][i]) + scaling_factor * (pop[r1][i] - pop[r2][i]);
      }
      else
      {
        child[i] = pop[target][i];
      }
    }
  }

  //If the mutant vector violates bounds, the bound handling method is applied
  modifySolutionWithParentMedium(child, pop[target]);
}

void S_LSHADE_DP::operateTarget1BinWithArchive(const vector<Individual> &pop, Individual child, int &target, variable &scaling_factor, variable &cross_rate, const vector<Individual> &archive, int &arc_ind_count)
{
  int r1, r2;

  do
  {
    r1 = rand() % pop_size;
  } while (r1 == target);
  do
  {
    r2 = rand() % (pop_size + arc_ind_count);
  } while ((r2 == target) || (r2 == r1));

  int random_variable = rand() % problem_size;

  if (r2 >= pop_size)
  {
    r2 -= pop_size;
    for (int i = 0; i < problem_size; i++)
    {
      if ((randDouble() < cross_rate) || (i == random_variable))
      {
        child[i] = pop[target][i] + scaling_factor * (pop[r1][i] - archive[r2][i]);
      }
      else
      {
        child[i] = pop[target][i];
      }
    }
  }
  else
  {
    for (int i = 0; i < problem_size; i++)
    {
      if ((randDouble() < cross_rate) || (i == random_variable))
      {
        child[i] = pop[target][i] + scaling_factor * (pop[r1][i] - pop[r2][i]);
      }
      else
      {
        child[i] = pop[target][i];
      }
    }
  }

  //If the mutant vector violates bounds, the bound handling method is applied
  modifySolutionWithParentMedium(child, pop[target]);
}

void S_LSHADE_DP::reducePopulationWithSort(vector<Individual> &pop, vector<Fitness> &fitness, vector<int> &stagnation)
{
  int worst_ind;

  for (int i = 0; i < reduction_ind_num; i++)
  {
    worst_ind = 0;
    for (int j = 1; j < pop_size; j++)
    {
      if (fitness[j] > fitness[worst_ind])
        worst_ind = j;
    }

    pop.erase(pop.begin() + worst_ind);
    fitness.erase(fitness.begin() + worst_ind);
    stagnation.erase(stagnation.begin() + worst_ind);
    pop_size--;
  }
}

void S_LSHADE_DP::setSHADEParameters()
{
  arc_rate = g_arc_rate;
  arc_size = (int)round(pop_size * arc_rate);
  p_best_rate = g_p_best_rate;
  memory_size = g_memory_size;
}