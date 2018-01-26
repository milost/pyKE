#ifndef CORRUPT_H
#define CORRUPT_H
#include "Random.h"
#include "Triple.h"
#include "Reader.h"




INT corrupt_head(INT id, INT h, INT r)
{
	INT lef, rig;

	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig)
	{
		INT mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r)
			rig = mid;
		else
			lef = mid;
	}
	INT ll = rig;

	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig)
	{
		INT mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r)
			lef = mid;
		else
			rig = mid;
	}
	INT rr = lef;

	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t)
		return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1)
		return tmp + rr - ll + 1;

	lef = ll, rig = rr + 1;
	while (lef + 1 < rig)
	{
		INT mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

INT corrupt_tail(INT id, INT t, INT r)
{
	INT lef, rig;

	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig)
	{
		INT mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r)
			rig = mid;
		else
			lef = mid;
	}
	INT ll = rig;

	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig)
	{
		INT mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r)
			lef = mid;
		else
			rig = mid;
	}
	INT rr = lef;

	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h)
		return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1)
		return tmp + rr - ll + 1;

	lef = ll;
	rig = rr + 1;
	while (lef + 1 < rig)
	{
		INT mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}


INT corrupt_rel(INT id, INT h, INT t)
{
	INT lef, rig;

	lef = lefRel[h] - 1;
	rig = rigRel[h];
	while (lef + 1 < rig)
	{
		INT mid = (lef + rig) >> 1;
		if (trainRel[mid].t >= t)
			rig = mid;
		else
			lef = mid;
	}
	INT ll = rig;

	lef = lefRel[h];
	rig = rigRel[h] + 1;
	while (lef + 1 < rig)
	{
		INT mid = (lef + rig) >> 1;
		if (trainRel[mid].t <= t)
			lef = mid;
		else
			rig = mid;
	}
	INT rr = lef;

	INT tmp = rand_max(id, relationTotal - (rr - ll + 1));
	if (tmp < trainRel[ll].r)
		return tmp;
	if (tmp > trainRel[rr].r - rr + ll - 1)
		return tmp + rr - ll + 1;

	lef = ll, rig = rr + 1;
	while (lef + 1 < rig)
	{
		INT mid = (lef + rig) >> 1;
		if (trainRel[mid].r - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

#endif
