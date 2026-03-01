/*
<SerializableBase.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Eremey Valetov

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.
The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef SERIALIZABLE_BASE_H
#define SERIALIZABLE_BASE_H

#include <serial/eoSerial.h>
#include <string>

/**
 * Wrapper that makes any type T serializable via ParadisEO's eoserial framework.
 * Used by MPI_IslandModel to serialize/deserialize populations for inter-process transfer.
 */
template<class T>
class SerializableBase : public eoserial::Persistent {
public:
    SerializableBase();
    SerializableBase(T base);
    virtual ~SerializableBase();

    operator T&();
    void setValue(const T& newValue);

    void unpack(const eoserial::Object* obj) override;
    eoserial::Object* pack() const override;

private:
    T _value;
};

#include "SerializableBase.tpp"

#endif // SERIALIZABLE_BASE_H
