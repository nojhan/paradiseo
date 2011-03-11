/***************************************************************************
 *   Copyright (C) 2008 by Waldo Cancino   *
 *   wcancino@icmc.usp.br   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#ifndef UTILS_H
#define UTILS_H

#include <PhyloMOEO.h>
#include <PhyloMOEO_init.h>
#include <likelihoodcalculator.h>
#include <ExceptionManager.h>
#include <sys/time.h>

void welcome_message();
void save_exp_params(ostream &);
void optimize_solutions( eoPop<PhyloMOEO> &);
void optimize_solution( PhyloMOEO &);
void readtrees(const char *, eoPop<PhyloMOEO> &);
int  timeval_subtract (struct timeval *, struct timeval *, struct timeval *);
void print_cpu_time(clock_t ,clock_t);
void print_elapsed_time(struct timeval *, struct timeval *);
void print_elapsed_time_short(struct timeval *, struct timeval *, ostream &os=cout);
#endif

